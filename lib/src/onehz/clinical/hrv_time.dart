// CLINICAL TIER-1 — time-domain HRV (PRV).
//
// Task Force 1996 conventions: RMSSD, SDNN, SDANN, pNN50, computed on the
// CLEANED NN series (run correctRr first). Window conventions:
//   ultra-short  : < 5 min   (RMSSD only, with caution)
//   short        : 5 min
//   24-h         : SDANN / SDNN-index use 5-min segment means / SDs.
//
// HONESTY: this is PRV (pulse-rate variability), not ECG HRV. RMSSD and pNNx
// are the metrics most biased by the 1 Hz beat-time quantization (successive-
// difference inflation) — flagged in `note`. Lead with SDNN / SDANN.
//
// That warning used to be advice only: every RMSSD in this file shipped, at
// confidence 0.95, however much of it was beat-timing jitter. [kNnDiffAcf1Floor]
// makes it behaviour as a cheap SCREEN; when it trips, [nnJitter]'s beat-indexed
// spectrum ARBITRATES, because respiratory sinus arrhythmia at a low heart rate
// trips the screen too (see [kNnDiffAcf1Floor] and `_judgeJitter`).

import 'dart:math' as math;
import 'dart:typed_data';
import '../types.dart';
import '../util.dart';
import '../respiration/resp_rate.dart' show respLoHz, respHiHz;

/// Lag-1 ACF floor for the NN successive-difference series, below which RMSSD
/// and pNN50 are REFUSED unless the spectrum shows the differences are
/// breathing (SDNN / SDANN survive either way and are the honest lead).
///
/// Differencing white noise leaves ACF1 at exactly −0.5, and a tachogram whose
/// differences are uncorrelated leaves it near 0; with a white share s of the
/// MSSD on top of such physiology, ACF1 = −s/2, so −0.35 means "≥ 70 % jitter".
/// That premise does NOT hold for respiratory sinus arrhythmia: RSA is a
/// sinusoid in beat index at b = breaths per beat, and its differences have
/// ACF1 = cos(2π·b) — below −0.35 once b > 0.307 (HR 48 at 16 br/min, HR 45 at
/// 14), and exactly −0.5, the white-noise value, at b = 1/3. ACF1 is therefore
/// only the cheap SCREEN; when it trips, `_judgeJitter` asks [nnJitter] for the
/// jitter share directly and keeps a night whose differences are one breathing
/// peak over a floor carrying ≤ [kJitterShareCeiling] of the MSSD (with a
/// margin for the estimate's spread), measured over ≥ [kJitterMinCoverage] of
/// the difference power. It is
/// deliberately the gate INSTEAD of a per-family constant
/// (`device.dart`): the sensor difference is real and large, but it reaches us
/// as something measurable, not as a label — a strap that starts reporting
/// cleaner beats is believed the night it does so, and an unknown strap is
/// judged on its own signal rather than refused for its badge.
///
/// MEASURED over the 13-night audit corpus: gen4 −0.057..−0.324,
/// MG −0.426..−0.456, WHOOP 5 −0.428..−0.517. −0.35 keeps every gen4 night and
/// refuses every gen5/MG night. OURS, not a published threshold — there is no
/// literature constant for this, and it is calibration, so it is a knob.
const double kNnDiffAcf1Floor = -0.35;

/// Fewest successive differences [nnDiffAcf1] will judge a series on. Below it
/// the ACF1 estimate is noisier than what it is meant to screen out, so it
/// returns null and NOTHING is gated — thin windows are already handled by the
/// beat-count term in confidence.
const int _acf1MinDiffs = 30;

/// Lag-1 autocorrelation of the successive-difference series, pooled over
/// CONTIGUOUS runs.
///
/// Each entry of [diffRuns] must be differences between beats that are adjacent
/// in time; a dropped run / sensor hole ends one run and starts the next, so no
/// lag-1 pair is ever formed across a seam. Null when there is too little to
/// judge or the series is constant.
double? nnDiffAcf1(List<List<double>> diffRuns) {
  var n = 0;
  var sum = 0.0;
  for (final r in diffRuns) {
    for (final d in r) {
      sum += d;
      n++;
    }
  }
  if (n < _acf1MinDiffs) return null;
  final m = sum / n;
  var cov = 0.0;
  var varSum = 0.0;
  for (final r in diffRuns) {
    for (var i = 0; i < r.length; i++) {
      final a = r[i] - m;
      varSum += a * a;
      if (i > 0) cov += (r[i - 1] - m) * a;
    }
  }
  return varSum > 0 ? cov / varSum : null;
}

/// Confidence multiplier for a measured [acf1]: 1.0 on a smooth tachogram,
/// falling linearly to 0 at [kNnDiffAcf1Floor] so confidence bottoms out
/// exactly where RMSSD is refused. 1.0 when ACF1 could not be measured.
double _acf1Quality(double? acf1) =>
    acf1 == null ? 1.0 : (1 - acf1 / kNnDiffAcf1Floor).clamp(0.0, 1.0);

String _pct(double x) => (100 * x).toStringAsFixed(1);

/// Why the spectrum could not rescue a night the ACF1 screen refused.
enum _JitterRefusal { evidence, noise, alternation, outOfBand }

String _jitterNote(double acf1, NnJitter? j, _JitterRefusal why) {
  final head = 'rmssd_refused:acf1=${acf1.toStringAsFixed(3)}'
      '${j == null ? '' : ',jitter_share=${j.share.toStringAsFixed(3)}'}';
  final body = switch (why) {
    _JitterRefusal.evidence => j == null
        ? 'the NN successive differences fail the jitter screen (floor '
            '$kNnDiffAcf1Floor) and there are too few contiguous beats for the '
            'beat-indexed spectrum ($kJitterMinSegments 64-beat segments) to '
            'tell breathing from beat-timing jitter, so the refusal stands'
        : 'the NN successive differences fail the jitter screen (floor '
            '$kNnDiffAcf1Floor) and the spectrum saw only ${_pct(j.coverage)} % '
            'of their power (it needs ${_pct(kJitterMinCoverage)} %; the rest '
            'sits in runs too short to assess), so the refusal stands',
    _JitterRefusal.noise =>
      'a broadband jitter floor carries ${_pct(j!.worstShare)} % of the NN '
          'successive-difference power (up to ${_pct(j.upperShare)} % within '
          'the estimate\'s spread; ceiling ${_pct(kJitterShareCeiling)} %), so '
          'RMSSD/pNN50 would measure beat-timing jitter, not vagal tone',
    _JitterRefusal.alternation =>
      'the NN successive differences fail the jitter screen and their '
          'dominant structure is a beat-to-beat alternation '
          '(${j!.peakCpb.toStringAsFixed(3)} cycles/beat) — ectopy or a '
          'detector artefact, not resolvable breathing',
    _JitterRefusal.outOfBand =>
      'the NN successive differences fail the jitter screen and their '
          'dominant peak'
          '${j!.peakHz == null ? '' : ' (${(j.peakHz! * 60).toStringAsFixed(1)} br/min)'}'
          ' lies outside the ${(respLoHz * 60).round()}–'
          '${(respHiHz * 60).round()} br/min breathing band, so it is not '
          'respiratory sinus arrhythmia',
  };
  return '$head — $body';
}

String _rsaKeptNote(double acf1, NnJitter j) =>
    'ACF1 ${acf1.toStringAsFixed(3)} is below the jitter floor, but the '
    'successive differences are a single respiratory peak '
    '(${(j.peakHz! * 60).toStringAsFixed(1)} br/min) over a jitter floor '
    'carrying ${_pct(j.worstShare)} % of the MSSD (at most '
    '${_pct(j.upperShare)} %) — RSA, not jitter; RMSSD kept';

/// Beats per Welch segment for [nnJitter]'s beat-indexed spectrum, and the hop
/// (50 % overlap, Welch 1967). Indexed by BEAT, not time: jitter is per beat and
/// RMSSD is a per-beat statistic, so cycles-per-beat is the natural axis.
const int _jitSegBeats = 64;
const int _jitSegStep = 32;

/// Fewest Welch segments [nnJitter] will judge on (~544 contiguous beats, ~9–11
/// min asleep). Below it the floor estimate is too noisy to overrule the screen
/// (white noise at 8 segments: share p1 0.72; at 17: p1 0.81), so the ACF1
/// verdict stands unchanged.
const int kJitterMinSegments = 16;

/// The largest jitter share of the MSSD a night the ACF1 screen refused may
/// carry and still be rescued: physiology must carry the MAJORITY of what is
/// published. Under the screen's own premise −0.35 meant a 70 % white share,
/// but that premise is exactly what fails here, and a rescue overturns a
/// refusal, so it holds itself to the stricter, symmetric line: more jitter
/// than physiology is never published as vagal tone.
const double kJitterShareCeiling = 0.5;

/// Fewest share of the successive-difference power (Σd²) [nnJitter]'s spectrum
/// must have SEEN before it may overrule the screen. Runs shorter than one
/// 64-beat segment, and the tails of longer ones, are invisible to it; their
/// power is charged as jitter in the worst case ([NnJitter.worstShare]), and
/// below 90 % assessed the rescue is refused outright — a spectrum that saw
/// a tenth or less of the evidence is not the judge of the rest.
const double kJitterMinCoverage = 0.9;

/// Search band for the dominant HF peak and the floor, cycles/beat. Below 0.1
/// the LF/VLF physiology lives; differencing barely passes it anyway.
const double _jitLoCpb = 0.1;

/// Bins excluded either side of the dominant peak when reading the floor.
const int _jitPeakHalfWidth = 3;

/// Lowest bin (0.25 cycles/beat) of the band the beat-TIME jitter model is fit
/// on: there sin²(π·f) ≥ 0.5, so the fit is not dominated by dividing small
/// low-frequency (LF/VLF) physiology by a near-zero model.
const int _jitTimingLoBin = 16;

/// A peak above this sits in the top two bins (≥ 0.47 cycles/beat): a
/// beat-to-beat ALTERNATION (detector artefact, bigeminy), not resolvable RSA.
/// Same reasoning as rsaRespRate refusing a peak at its own ceiling.
const double _jitMaxPeakCpb = 0.47;

/// The night's beat-timing jitter, measured from the spectrum's SHAPE.
class NnJitter {
  /// Estimated fraction of the ASSESSED MSSD explained by a jitter floor, 0..1:
  /// the larger of the white-NN and the beat-time-jitter model.
  final double share;

  /// Fraction of the successive-difference power (Σd²) inside the spectrum's
  /// segments, 0..1.
  final double coverage;

  /// Dominant peak in [0.1, 0.5] cycles/beat.
  final double peakCpb;

  /// The same peak in Hz (peakCpb / median NN in s); null if NN is degenerate.
  final double? peakHz;
  final int segments;
  const NnJitter({
    required this.share,
    required this.coverage,
    required this.peakCpb,
    this.peakHz,
    required this.segments,
  });

  /// The jitter share of ALL the power, charging every unassessed difference
  /// as jitter: `coverage·share + (1 − coverage)`.
  double get worstShare => coverage * share + (1 - coverage);

  /// [worstShare] plus a one-sided 99 % margin for the estimate's own spread:
  /// `2.33 · 0.25 / √segments`. MEASURED (200 draws per point, white and
  /// beat-time jitter at a true share of 0.5): the estimate's SD × √segments
  /// is 0.18–0.23 from 16 to 200 segments, so 0.25 is a rounded-up bound. The
  /// rescue judges this, so a night that is truly half jitter is not rescued
  /// by a lucky draw — at 16 segments the margin is 0.15, over a whole night
  /// (~900 segments) 0.02.
  double get upperShare =>
      worstShare + 2.33 * 0.25 / math.sqrt(segments.toDouble());
  Map<String, dynamic> toJson() => {
        'jitter_share': round6(share),
        'jitter_coverage': round6(coverage),
        'jitter_peak_cpb': round6(peakCpb),
        if (peakHz != null) 'jitter_peak_hz': round6(peakHz!),
        'jitter_segments': segments,
      };
}

const int _jitBins = _jitSegBeats ~/ 2 + 1;
final Float64List _jitHann = Float64List.fromList([
  for (var k = 0; k < _jitSegBeats; k++)
    0.5 - 0.5 * math.cos(2 * math.pi * k / (_jitSegBeats - 1))
]);
// DFT kernel, row k = bin, column j = beat: cos/sin(2π·k·j/64).
final Float64List _jitCos = Float64List.fromList([
  for (var k = 0; k < _jitBins; k++)
    for (var j = 0; j < _jitSegBeats; j++)
      math.cos(2 * math.pi * ((k * j) % _jitSegBeats) / _jitSegBeats)
]);
final Float64List _jitSin = Float64List.fromList([
  for (var k = 0; k < _jitBins; k++)
    for (var j = 0; j < _jitSegBeats; j++)
      math.sin(2 * math.pi * ((k * j) % _jitSegBeats) / _jitSegBeats)
]);

/// White (beat-timing) share of the MSSD, from a beat-indexed Welch spectrum.
///
/// White NN noise of variance σ² puts power σ² at EVERY beat frequency up to
/// the beat Nyquist (0.5 cycles/beat) and contributes exactly 2σ² to the MSSD;
/// beat-TIME jitter is broadband too. Respiratory sinus arrhythmia is one
/// narrow peak at the breathing frequency (Hirsch & Bishop 1981), and timing
/// quantisation adds a broadband floor (Merri et al. 1990). The jitter share
/// is read from the spectrum's SHAPE, which no single autocorrelation lag can
/// do (at b = 1/3 pure RSA and pure noise share ACF1 = −0.5), under the two
/// noise shapes a beat stream carries, and the LARGER is kept:
///   * white NN noise (variance σ²): flat PSD σ², MSSD share 2σ² —
///     `2·median(PSD over 0.1–0.5 cycles/beat, peak ±3 bins excluded)`;
///   * beat-TIME jitter (each timestamp off by an independent σ): NN noise is
///     its first difference, PSD 4σ²·sin²(π·f), MSSD share 6σ² —
///     `6·median(PSD/(4·sin²(π·f)) over 0.25–0.5 cycles/beat, peak excluded)`.
///     The white model alone reads this shape low (measured 56 % for a true
///     79 %), because the floor it sees is depressed below the band's middle.
///
/// [nnRuns] are contiguous LEVEL runs (NN values of beats adjacent in time — a
/// seam starts a new run). Welch 1967: 64-beat Hann segments, 50 % overlap,
/// each linearly detrended; a segment never crosses a run boundary. The share
/// is over the MSSD of exactly the differences inside those segments, and
/// [NnJitter.coverage] says how much of ALL the within-run Σd² that is — runs
/// too short for a segment are invisible to the spectrum and must not dilute
/// it. Null when fewer than [kJitterMinSegments] segments fit, or the assessed
/// MSSD is zero.
NnJitter? nnJitter(List<List<double>> nnRuns) {
  const nb = _jitBins;
  final acc = Float64List(nb);
  var w2 = 0.0;
  for (final w in _jitHann) {
    w2 += w * w;
  }
  const tb = (_jitSegBeats - 1) / 2.0;
  var tt = 0.0;
  for (var i = 0; i < _jitSegBeats; i++) {
    tt += (i - tb) * (i - tb);
  }
  final y = Float64List(_jitSegBeats);
  var segments = 0;
  var ssdAll = 0.0;
  var ssd = 0.0; // inside the segments only
  var nd = 0;
  final levels = <double>[];
  for (final run in nnRuns) {
    levels.addAll(run);
    // Beats [0, seen) are covered by this run's segments.
    final fit = run.length - _jitSegBeats;
    final seen = fit < 0 ? 0 : fit - fit % _jitSegStep + _jitSegBeats;
    for (var i = 1; i < run.length; i++) {
      final d = run[i] - run[i - 1];
      ssdAll += d * d;
      if (i < seen) {
        ssd += d * d;
        nd++;
      }
    }
    for (var s = 0; s + _jitSegBeats <= run.length; s += _jitSegStep) {
      var mu = 0.0;
      for (var i = 0; i < _jitSegBeats; i++) {
        mu += run[s + i];
      }
      mu /= _jitSegBeats;
      var sl = 0.0;
      for (var i = 0; i < _jitSegBeats; i++) {
        sl += (i - tb) * (run[s + i] - mu);
      }
      sl /= tt;
      for (var i = 0; i < _jitSegBeats; i++) {
        y[i] = (run[s + i] - mu - sl * (i - tb)) * _jitHann[i];
      }
      for (var k = 0; k < nb; k++) {
        var re = 0.0, im = 0.0;
        final row = k * _jitSegBeats;
        for (var j = 0; j < _jitSegBeats; j++) {
          re += y[j] * _jitCos[row + j];
          im += y[j] * _jitSin[row + j];
        }
        acc[k] += (re * re + im * im) / w2;
      }
      segments++;
    }
  }
  if (segments < kJitterMinSegments || nd == 0) return null;
  final mssd = ssd / nd;
  if (mssd <= 0) return null;
  final psd = [for (final a in acc) a / segments];
  final kLo = (_jitLoCpb * _jitSegBeats).ceil();
  const kHi = _jitSegBeats ~/ 2;
  var kPk = kLo;
  for (var k = kLo + 1; k <= kHi; k++) {
    if (psd[k] > psd[kPk]) kPk = k;
  }
  final floorBins = [
    for (var k = kLo; k <= kHi; k++)
      if ((k - kPk).abs() > _jitPeakHalfWidth) psd[k]
  ];
  final white = 2 * median(floorBins)!;
  final timingBins = [
    for (var k = _jitTimingLoBin; k <= kHi; k++)
      if ((k - kPk).abs() > _jitPeakHalfWidth)
        psd[k] / (4 * math.pow(math.sin(math.pi * k / _jitSegBeats), 2))
  ];
  final timing = timingBins.isEmpty ? 0.0 : 6 * median(timingBins)!;
  final peakCpb = kPk / _jitSegBeats;
  final medNn = median(levels);
  return NnJitter(
    share: (math.max(white, timing) / mssd).clamp(0.0, 1.0),
    coverage: ssdAll > 0 ? (ssd / ssdAll).clamp(0.0, 1.0) : 1.0,
    peakCpb: peakCpb,
    peakHz: (medNn != null && medNn > 0) ? peakCpb / (medNn / 1000.0) : null,
    segments: segments,
  );
}

/// One jitter verdict, shared by every RMSSD in this file.
class _JitterVerdict {
  final bool refused;

  /// Confidence multiplier; [_acf1Quality] whenever the screen does not trip.
  final double quality;

  /// Diagnostic, emitted whenever measurable.
  final NnJitter? jitter;
  final String? note;
  const _JitterVerdict(this.refused, this.quality, this.jitter, this.note);
}

/// ACF1 screens; the spectrum arbitrates only what the screen would refuse, so
/// the set of refused nights can only SHRINK and a night at ACF1 ≥ −0.35 is
/// judged exactly as before. Kept only when the spectrum saw ≥
/// [kJitterMinCoverage] of the difference power, the worst-case jitter share
/// (with its 99 % margin, [NnJitter.upperShare]) is ≤ [kJitterShareCeiling],
/// and the dominant peak is breathing
/// ([respLoHz]–[respHiHz], below [_jitMaxPeakCpb]); otherwise refused, with the
/// cause in the note.
_JitterVerdict _judgeJitter(double? acf1, List<List<double>> nnRuns) {
  final j = nnJitter(nnRuns);
  if (acf1 == null || acf1 >= kNnDiffAcf1Floor) {
    return _JitterVerdict(false, _acf1Quality(acf1), j, null);
  }
  final _JitterRefusal? why;
  if (j == null || j.coverage < kJitterMinCoverage) {
    why = _JitterRefusal.evidence;
  } else if (j.upperShare > kJitterShareCeiling) {
    why = _JitterRefusal.noise;
  } else if (j.peakCpb > _jitMaxPeakCpb) {
    why = _JitterRefusal.alternation;
  } else if (j.peakHz == null ||
      j.peakHz! < respLoHz ||
      j.peakHz! > respHiHz) {
    why = _JitterRefusal.outOfBand;
  } else {
    why = null;
  }
  if (why == null) {
    return _JitterVerdict(
      false,
      (1 - j!.worstShare / kJitterShareCeiling).clamp(0.0, 1.0),
      j,
      _rsaKeptNote(acf1, j),
    );
  }
  return _JitterVerdict(true, 0.0, j, _jitterNote(acf1, j, why));
}

/// Σ reported RR ÷ elapsed wall time above which the stream cannot be one
/// heart's beats (it banks more beat-time than time passed): a double ingest or
/// two interleaved streams. Contiguous runs measure 0.963 (gen4), 0.999 (W5),
/// 1.001 (MG) — see rr_correction.dart `_beatTimes` — so 1.10 has margin.
/// Duplicated beats add zero differences and DEFLATE RMSSD by ~1/√2;
/// interleaved streams inflate it. Every other gate here passes both.
const double kRrCoverageCeiling = 1.10;

/// Shortest wall span [rrCoverage] will judge: whole-second stamps make a
/// shorter one meaningless.
const double kRrCoverageMinSpanSec = 600;

/// How much beat-time an RR stream banks against the wall clock it spans.
class RrCoverage {
  /// Σ plausible RR ÷ wall span. Below 1 on any gap; above 1 is impossible
  /// for one heart.
  final double coverage;
  final double sumRrSec;
  final double spanSec;
  final int beats;

  /// Intervals outside [300, 2400] ms: counted, never summed.
  final int implausibleBeats;

  /// Exact (ts, rr) repeats of the previous beat. A DIAGNOSTIC only: on
  /// whole-second stamps two equal beats in one record repeat legitimately.
  final int duplicateBeats;
  const RrCoverage({
    required this.coverage,
    required this.sumRrSec,
    required this.spanSec,
    required this.beats,
    required this.implausibleBeats,
    required this.duplicateBeats,
  });
  bool get overCounted => coverage > kRrCoverageCeiling;
  Map<String, dynamic> toJson() => {
        'rr_coverage': round6(coverage),
        'sum_rr_sec': round6(sumRrSec),
        'span_sec': round6(spanSec),
        'beats': beats,
        'implausible_beats': implausibleBeats,
        'duplicate_beats': duplicateBeats,
      };
}

/// [RrCoverage] of raw RR [rrMs] against their beat-END epoch times [rrTsMs]
/// (time-sorted, same length). The span is `last − first + rrMs.first`: beat 0
/// began before its end stamp. Null when fewer than 2 beats, the lengths
/// differ, or the span is under [kRrCoverageMinSpanSec].
RrCoverage? rrCoverage(List<double> rrMs, List<double> rrTsMs) {
  if (rrMs.length < 2 || rrMs.length != rrTsMs.length) return null;
  final spanSec = (rrTsMs.last - rrTsMs.first + rrMs.first) / 1000.0;
  if (!spanSec.isFinite || spanSec < kRrCoverageMinSpanSec) return null;
  var sum = 0.0;
  var implausible = 0;
  var dup = 0;
  for (var i = 0; i < rrMs.length; i++) {
    final v = rrMs[i];
    if (v >= 300 && v <= 2400) {
      sum += v;
    } else {
      implausible++;
    }
    if (i > 0 && v == rrMs[i - 1] && rrTsMs[i] == rrTsMs[i - 1]) dup++;
  }
  final sumSec = sum / 1000.0;
  return RrCoverage(
    coverage: sumSec / spanSec,
    sumRrSec: sumSec,
    spanSec: spanSec,
    beats: rrMs.length,
    implausibleBeats: implausible,
    duplicateBeats: dup,
  );
}

String _overcountNote(RrCoverage c) =>
    'rr_overcount:coverage=${round6(c.coverage)} — more beat-time than '
    'elapsed time; the RR stream holds duplicated or interleaved beats';

class HrvTime {
  final double? rmssd; // ms
  final double? sdnn; // ms
  final double? sdann; // ms (24-h: SD of 5-min means)
  final double? sdnnIndex; // ms (24-h: mean of 5-min SDs)
  final double? pnn50; // %
  final int nBeats;
  final double? diffAcf1; // lag-1 ACF of the NN successive differences
  final double? jitterShare; // [nnJitter] white share of the MSSD
  const HrvTime({
    this.rmssd,
    this.sdnn,
    this.sdann,
    this.sdnnIndex,
    this.pnn50,
    required this.nBeats,
    this.diffAcf1,
    this.jitterShare,
  });
  Map<String, dynamic> toJson() => {
        if (rmssd != null) 'rmssd_ms': round6(rmssd!),
        if (sdnn != null) 'sdnn_ms': round6(sdnn!),
        if (sdann != null) 'sdann_ms': round6(sdann!),
        if (sdnnIndex != null) 'sdnn_index_ms': round6(sdnnIndex!),
        if (pnn50 != null) 'pnn50_pct': round6(pnn50!),
        'n_beats': nBeats,
        if (diffAcf1 != null) 'diff_acf1': round6(diffAcf1!),
        if (jitterShare != null) 'jitter_share': round6(jitterShare!),
      };
}

/// Short-window time-domain HRV on a cleaned NN series (ms).
///
/// [nnMs] cleaned NN intervals. [nnTimesMs] beat times, used both for
/// SDANN/SDNN-index segmentation and to skip successive-difference pairs that
/// straddle a dropped run (optional; without it SDANN/SDNN-index are null and
/// RMSSD/pNN50 include the seams). [artifactFraction] is the fraction of beats
/// the upstream corrector rejected (0..1), folded into confidence exactly as
/// `hrvFreq` and `irregularBeatScreen` already do. Returns an absent Metric when
/// there are too few beats; RMSSD/pNN50 alone go null when the successive
/// differences fail [kNnDiffAcf1Floor] and the spectrum does not show them to
/// be breathing (`_judgeJitter`), or when [coverage] (of the raw RR this NN
/// was cleaned from) is [RrCoverage.overCounted]. Without [coverage] that last
/// check is skipped.
Metric<HrvTime> hrvTime(
  List<double> nnMs, {
  List<double>? nnTimesMs,
  double artifactFraction = 0.0,
  RrCoverage? coverage,
}) {
  const inputs = ['rr_cleaned'];
  if (nnMs.length < 2) {
    return const Metric<HrvTime>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'too few NN intervals',
    );
  }

  // RMSSD / pNN50: root mean square of SUCCESSIVE differences — successive in
  // TIME, not merely adjacent in the compacted list. correctRr drops multi-beat
  // artifact runs while advancing its clock across them, so nn[i-1] and nn[i]
  // can sit either side of a seconds-long hole; differencing straight down the
  // list manufactured one large difference per dropped run. Same `keep`-mask
  // treatment irregular_rhythm.dart already applies. A pair is contiguous iff
  // the elapsed time between the two beat times is the interval itself.
  //
  // The differences are kept as contiguous RUNS (a seam ends a run) so the same
  // pass feeds [nnDiffAcf1] without ever forming a lag-1 pair across a hole.
  // The same seams cut the LEVEL runs [nnJitter] reads, so no Welch segment
  // spans a dropout either.
  final gapAware = nnTimesMs != null && nnTimesMs.length == nnMs.length;
  final runs = <List<double>>[];
  final levelRuns = <List<double>>[];
  var run = <double>[];
  var level = <double>[nnMs[0]];
  for (var i = 1; i < nnMs.length; i++) {
    if (gapAware && nnTimesMs[i] - nnTimesMs[i - 1] > nnMs[i] + 0.5) {
      if (run.isNotEmpty) {
        runs.add(run);
        run = <double>[];
      }
      levelRuns.add(level);
      level = <double>[nnMs[i]];
      continue;
    }
    run.add(nnMs[i] - nnMs[i - 1]);
    level.add(nnMs[i]);
  }
  if (run.isNotEmpty) runs.add(run);
  levelRuns.add(level);

  var ssd = 0.0;
  var nn50 = 0;
  var pairs = 0;
  for (final r in runs) {
    for (final d in r) {
      ssd += d * d;
      if (d.abs() > 50) nn50++;
      pairs++;
    }
  }
  // RMSSD and pNN50 are the two outputs made of successive differences, so they
  // are the two the jitter floor refuses. SDNN/SDANN are made of the levels and
  // are far less contaminated (jitter share 1–27 % against RMSSD's 11–100 % on
  // the audit corpus) — they keep publishing, which is what the header has
  // always advised.
  final acf1 = nnDiffAcf1(runs);
  final verdict = _judgeJitter(acf1, levelRuns);
  final overCounted = coverage?.overCounted == true;
  final jittery = verdict.refused || overCounted;
  final rmssd = (pairs > 0 && !jittery) ? math.sqrt(ssd / pairs) : null;
  final pnn50 = (pairs > 0 && !jittery) ? 100.0 * nn50 / pairs : null;
  final sdnn = stddev(nnMs);

  double? sdann, sdnnIndex;
  if (gapAware) {
    final seg = _fiveMinSegments(nnMs, nnTimesMs);
    if (seg.length >= 2) {
      final means = [for (final s in seg) mean(s)!];
      sdann = stddev(means);
      final sds = [for (final s in seg) stddev(s)].whereType<double>().toList();
      sdnnIndex = sds.isEmpty ? null : mean(sds);
    }
  }

  // Confidence scales with beat count (ultra-short reads are less reliable),
  // with the artifact fraction we were handed, and with the measured jitter
  // level. It used to be beat count alone, which published 0.95 on all 13 nights
  // of the audit corpus — including a 15.3 %-artifact night whose differences
  // were ~pure noise. The beat-count term is capped BEFORE the quality terms
  // multiply it; multiplying first let an all-night beat count (n/250 ≈ 100)
  // swallow any penalty and re-clamp to 0.95 regardless.
  final conf = ((nnMs.length / 250.0).clamp(0.0, 1.0) // ~250 beats ≈ 5 min
          *
          (overCounted ? 0.0 : verdict.quality) *
          (1 - artifactFraction))
      .clamp(0.3, 0.95);
  return Metric<HrvTime>(
    value: HrvTime(
      rmssd: rmssd,
      sdnn: sdnn,
      sdann: sdann,
      sdnnIndex: sdnnIndex,
      pnn50: pnn50,
      nBeats: nnMs.length,
      diffAcf1: acf1,
      jitterShare: verdict.jitter?.share,
    ),
    confidence: conf,
    tier: Tier.high,
    inputs_used: inputs,
    note: jittery
        ? '${overCounted ? _overcountNote(coverage!) : verdict.note}. '
            'SDNN/SDANN survive it and are the lead here. PRV not ECG-HRV.'
        : 'PRV not ECG-HRV; RMSSD/pNN50 are quantization-sensitive at 1 Hz '
            '— lead with SDNN/SDANN'
            '${verdict.note == null ? '' : '. ${verdict.note}'}',
  );
}

/// Robust NOCTURNAL RMSSD (ms).
///
/// A single whole-night RMSSD is dominated by the few high-Δ segments produced
/// by REM bursts, arousals and stage transitions, inflating it well above the
/// resting parasympathetic level (~tens of ms). Instead we compute RMSSD WITHIN
/// each consecutive ~5-min window of the NN series and take the MEDIAN across
/// windows — a robust estimator far less sensitive to a handful of high-variance
/// windows. Optionally restrict to NREM / low-motion windows via [stageMaskPerSec].
///
/// [nnMs] cleaned NN intervals. [nnTimesMs] beat times (ms, same length) used to
/// window into 5-min bins; required (returns absent without it). [windowMs] bin
/// width (default 300 000 = 5 min). [minBeatsPerWindow] min NN diffs a window
/// needs to contribute (default 5). [stageMaskPerSec] OPTIONAL per-second mask
/// (true = keep, e.g. NREM & immobile); a window is kept only when the mask is
/// true at the window's MIDPOINT second.
///
/// Returns a Metric whose value is the median-of-windows RMSSD (ms). Keeps the
/// PRV-not-ECG honesty note. Absent when there are too few usable windows, or
/// when the night's successive differences fail [kNnDiffAcf1Floor] and the
/// spectrum does not show them to be breathing (`_judgeJitter`), or when
/// [coverage] (of the raw RR) is [RrCoverage.overCounted]. A window
/// contributes only if it holds [minBeatsPerWindow] differences between beats
/// that are ADJACENT IN TIME, not merely adjacent in the compacted NN list.
Metric<double> nocturnalRmssd(
  List<double> nnMs,
  List<double> nnTimesMs, {
  double windowMs = 300000.0,
  int minBeatsPerWindow = 5,
  List<bool>? stageMaskPerSec,
  RrCoverage? coverage,
}) {
  const inputs = ['rr_cleaned', 'beat_times'];
  if (coverage != null && coverage.overCounted) {
    return Metric<double>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: _overcountNote(coverage),
    );
  }
  if (nnMs.length != nnTimesMs.length || nnMs.length < minBeatsPerWindow + 1) {
    return const Metric<double>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'too few NN intervals for windowed nocturnal RMSSD',
    );
  }
  final t0 = nnTimesMs.first;
  // Bucket beat INDICES by window index, so each window keeps its beat times and
  // the successive differences below can skip the ones that straddle a dropped
  // run — the same seam rule `hrvTime` applies.
  final buckets = <int, List<int>>{};
  for (var i = 0; i < nnMs.length; i++) {
    final idx = ((nnTimesMs[i] - t0) / windowMs).floor();
    (buckets[idx] ??= <int>[]).add(i);
  }
  // Compute per-window RMSSD over the windows we keep. Each window's difference
  // series is also kept as one contiguous run for the jitter floor below.
  final rmssds = <double>[];
  // The jitter floor is judged over the WHOLE night, pooled across windows, not
  // per window: at 5 min a window holds a few hundred differences and its ACF1
  // is noisy enough that dropping only the windows that fail keeps the ones that
  // passed by luck — measured, that let WHOOP 5 publish 109-116 ms from its
  // calmest-looking windows while the night pooled to −0.43/−0.51.
  final runs = <List<double>>[];
  final levelRuns = <List<double>>[]; // same seams, NN levels — [nnJitter]
  final indices = buckets.keys.toList()..sort();
  for (final idx in indices) {
    if (stageMaskPerSec != null) {
      final midSec = ((idx + 0.5) * windowMs / 1000.0).floor();
      final keep = midSec >= 0 &&
          midSec < stageMaskPerSec.length &&
          stageMaskPerSec[midSec];
      if (!keep) continue;
    }
    final seg = buckets[idx]!;
    if (seg.length < minBeatsPerWindow + 1) continue;
    // Contiguous runs inside the window: a pair whose beat times are further
    // apart than the interval itself sits either side of a dropped run, and
    // differencing across it manufactures one large difference per hole.
    final winRuns = <List<double>>[];
    final winLevels = <List<double>>[];
    var run = <double>[];
    var level = <double>[nnMs[seg[0]]];
    for (var k = 1; k < seg.length; k++) {
      final i = seg[k], p = seg[k - 1];
      if (nnTimesMs[i] - nnTimesMs[p] > nnMs[i] + 0.5) {
        if (run.isNotEmpty) {
          winRuns.add(run);
          run = <double>[];
        }
        winLevels.add(level);
        level = <double>[nnMs[i]];
        continue;
      }
      run.add(nnMs[i] - nnMs[p]);
      level.add(nnMs[i]);
    }
    if (run.isNotEmpty) winRuns.add(run);
    winLevels.add(level);
    var ssd = 0.0;
    var nd = 0;
    for (final r in winRuns) {
      for (final d in r) {
        ssd += d * d;
        nd++;
      }
    }
    if (nd < minBeatsPerWindow) continue;
    runs.addAll(winRuns);
    levelRuns.addAll(winLevels);
    rmssds.add(math.sqrt(ssd / nd));
  }
  final acf1 = nnDiffAcf1(runs);
  final verdict = _judgeJitter(acf1, levelRuns);
  if (verdict.refused) {
    return Metric<double>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: verdict.note,
    );
  }
  if (rmssds.isEmpty) {
    return const Metric<double>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'no usable 5-min windows for nocturnal RMSSD',
    );
  }
  final robust = median(rmssds)!;
  // Confidence scales with how many windows we could median over, and with the
  // measured jitter level (see [kNnDiffAcf1Floor]).
  final conf =
      ((rmssds.length / 12.0).clamp(0.0, 1.0) * verdict.quality).clamp(
          // 12 ≈ 1 h
          0.3,
          0.95);
  return Metric<double>(
    value: robust,
    confidence: conf,
    tier: Tier.high,
    inputs_used: inputs,
    note: 'robust nocturnal RMSSD = MEDIAN of ${rmssds.length} consecutive '
        '5-min-window RMSSDs (REM/arousal-robust). PRV not ECG-HRV; '
        'RMSSD is quantization-sensitive at 1 Hz.'
        '${verdict.note == null ? '' : ' ${verdict.note}.'}',
  );
}

/// The nightly headline RMSSD plus the diagnostics a `Metric<double>` cannot
/// carry. Present only when the headline is.
class SessionRmssd {
  final double rmssd; // ms — the headline
  final int windows; // 5-min windows that contributed
  final double? diffAcf1; // pooled over those windows
  final double? jitterShare; // [nnJitter] over the same windows
  final double? rrCoverage; // [RrCoverage.coverage] of the session's beats
  const SessionRmssd({
    required this.rmssd,
    required this.windows,
    this.diffAcf1,
    this.jitterShare,
    this.rrCoverage,
  });
  Map<String, dynamic> toJson() => {
        'rmssd_ms': round6(rmssd),
        'windows': windows,
        if (diffAcf1 != null) 'diff_acf1': round6(diffAcf1!),
        if (jitterShare != null) 'jitter_share': round6(jitterShare!),
        if (rrCoverage != null) 'rr_coverage': round6(rrCoverage!),
      };
}

/// Sleep-session nightly RMSSD (ms) as the arithmetic mean of cleaned
/// consecutive 5-minute window RMSSDs.
///
/// Split the detected sleep session into consecutive 5-minute windows, apply a
/// simple RR cleaner (range-filter [300, 2000] ms + Malik-style ectopic
/// rejection against a local median), compute RMSSD inside each valid window,
/// then return the ARITHMETIC MEAN across windows. This is intentionally
/// distinct from [nocturnalRmssd], which uses cleaned NN +
/// median-of-windows robustness.
///
/// This is the nightly HEADLINE (→ `ln_rmssd` → readiness), so it refuses
/// rather than approximates: absent when the successive differences fail
/// [kNnDiffAcf1Floor] and the spectrum does not show them to be breathing
/// (`_judgeJitter`), and absent when the session's beats bank more time than
/// elapsed ([kRrCoverageCeiling]).
///
/// [rrMs]/[rrTsMs] are the raw RR intervals and their beat-end epoch times in
/// milliseconds. [startSec]/[endSec] bound the chosen sleep session in epoch
/// seconds. The implementation is one-pass over the time-sorted RR stream:
/// beats are bucketed once by `(tsSec - startSec) ~/ windowSec`.
Metric<double> sleepSessionWindowedRmssd(
  List<double> rrMs,
  List<double> rrTsMs, {
  required int startSec,
  required int endSec,
  int windowSec = 300,
}) {
  final m = sleepSessionRmssdDetail(rrMs, rrTsMs,
      startSec: startSec, endSec: endSec, windowSec: windowSec);
  return m.present
      ? Metric<double>(
          value: m.value!.rmssd,
          confidence: m.confidence,
          tier: m.tier,
          inputs_used: m.inputs_used,
          note: m.note,
        )
      : Metric<double>.absent(
          tier: m.tier, inputs_used: m.inputs_used, note: m.note);
}

/// [sleepSessionWindowedRmssd] with its diagnostics ([SessionRmssd]). The two
/// are one computation; this is the one that does it.
Metric<SessionRmssd> sleepSessionRmssdDetail(
  List<double> rrMs,
  List<double> rrTsMs, {
  required int startSec,
  required int endSec,
  int windowSec = 300,
}) {
  const inputs = ['rr_sleep_window'];
  if (startSec <= 0 ||
      endSec <= startSec ||
      rrMs.isEmpty ||
      rrTsMs.isEmpty ||
      rrMs.length != rrTsMs.length) {
    return const Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'invalid or empty RR session window',
    );
  }

  final buckets = <int, List<double>>{};
  final bucketsTs = <int, List<double>>{};
  final inRr = <double>[];
  final inTs = <double>[];
  for (var i = 0; i < rrMs.length; i++) {
    final tsSec = (rrTsMs[i] / 1000.0).round();
    if (tsSec < startSec || tsSec >= endSec) continue;
    final idx = ((tsSec - startSec) ~/ windowSec);
    (buckets[idx] ??= <double>[]).add(rrMs[i]);
    (bucketsTs[idx] ??= <double>[]).add(rrTsMs[i]);
    inRr.add(rrMs[i]);
    inTs.add(rrTsMs[i]);
  }

  if (buckets.isEmpty) {
    return const Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'no RR beats inside the session window',
    );
  }
  final cov = rrCoverage(inRr, inTs);
  if (cov != null && cov.overCounted) {
    return Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: _overcountNote(cov),
    );
  }

  final rmssds = <double>[];
  final runs = <List<double>>[]; // pooled jitter floor — see [nocturnalRmssd]
  final levelRuns = <List<double>>[]; // the same windows' levels — [nnJitter]
  final indices = buckets.keys.toList()..sort();
  for (final idx in indices) {
    final cleanRuns = _cleanWindowRuns(buckets[idx]!, bucketsTs[idx]!);
    final diffRuns = [
      for (final r in cleanRuns)
        if (r.length >= 2) [for (var i = 1; i < r.length; i++) r[i] - r[i - 1]]
    ];
    var ssd = 0.0;
    var nd = 0;
    for (final r in diffRuns) {
      for (final d in r) {
        ssd += d * d;
        nd++;
      }
    }
    if (nd == 0) continue;
    runs.addAll(diffRuns);
    levelRuns.addAll(cleanRuns);
    rmssds.add(math.sqrt(ssd / nd));
  }

  // THE HEADLINE nightly RMSSD (→ ln_rmssd → readiness). When the differences
  // are noise, the honest output is no headline, not a plausible one — the
  // readiness composite already treats a null HRV driver as absent.
  final acf1 = nnDiffAcf1(runs);
  final verdict = _judgeJitter(acf1, levelRuns);
  if (verdict.refused) {
    return Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: verdict.note,
    );
  }
  if (rmssds.isEmpty) {
    return const Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'no valid 5-min windows for sleep-session RMSSD',
    );
  }

  final meanRmssd = mean(rmssds)!;
  final conf = ((rmssds.length / 12.0).clamp(0.0, 1.0) * verdict.quality)
      .clamp(0.3, 0.95);
  return Metric<SessionRmssd>(
    value: SessionRmssd(
      rmssd: meanRmssd,
      windows: rmssds.length,
      diffAcf1: acf1,
      jitterShare: verdict.jitter?.share,
      rrCoverage: cov?.coverage,
    ),
    confidence: conf,
    tier: Tier.high,
    inputs_used: inputs,
    note: 'sleep-session HRV: mean RMSSD over cleaned 5-min windows.'
        '${verdict.note == null ? '' : ' ${verdict.note}.'}',
  );
}

/// Group NN intervals into consecutive 5-minute (300 000 ms) segments by beat
/// time. Segments with <2 beats are dropped.
List<List<double>> _fiveMinSegments(List<double> nn, List<double> times) {
  const segMs = 300000.0;
  final out = <List<double>>[];
  if (nn.isEmpty) return out;
  final t0 = times.first;
  var curIdx = 0;
  var cur = <double>[];
  for (var i = 0; i < nn.length; i++) {
    final idx = ((times[i] - t0) / segMs).floor();
    if (idx != curIdx) {
      if (cur.length >= 2) out.add(cur);
      cur = <double>[];
      curIdx = idx;
    }
    cur.add(nn[i]);
  }
  if (cur.length >= 2) out.add(cur);
  return out;
}

/// Range-filter [300, 2000] ms + Malik-style ectopic rejection against a local
/// median, returned as CONTIGUOUS RUNS of kept intervals.
///
/// Runs, not one compacted list: differencing straight down a compacted list
/// manufactures exactly one difference per rejected beat, spanning it — the same
/// defect `hrvTime` refuses at dropped runs and `irregularBeatScreen` refuses
/// with its keep-mask. MEASURED over the 13-night audit corpus: it inflated
/// the headline by 2–13 % on gen4 (57.2 → 52.3 ms at worst) and by 51–102 % on
/// MG (87.7 → 58.2, 82.9 → 40.9, 76.9 → 40.2 ms) — i.e. most of the "gen5
/// reads 2× gen4" gap was this, not physiology.
///
/// [ts] are [rr]'s beat-end epoch times (ms), same length/order as [rr]. Also
/// breaks a run across a real sensor gap between two beats that BOTH survive
/// the range/median filter — the same seam check `nocturnalRmssd` applies via
/// `nnTimesMs`, needed here too since two beats either side of a dropout can
/// individually pass and land adjacent in the compacted survivor list.
///
/// The real caller (`_sessionAvgHRV`) quantizes [ts] to whole seconds
/// (`RrTs.ts` is `(rrTsMs / 1000.0).round()`), so two independent roundings
/// can disagree with the true interval by up to ~1000 ms with no dropout at
/// all — the tolerance is `nn[i] + 1000.0`, not `nocturnalRmssd`'s `+ 0.5`
/// (which assumes sub-second beat times), so quantization alone never trips
/// it while an actual multi-second-or-longer dropout still does.
List<List<double>> _cleanWindowRuns(List<double> rr, List<double> ts) {
  const radius = 2;
  const threshold = 0.20;
  // Range filter first, keeping each survivor's position in [rr] — BOTH filters
  // break a run, so neither one's compaction can manufacture a difference.
  final nn = <double>[];
  final at = <int>[];
  final nnTs = <double>[];
  for (var i = 0; i < rr.length; i++) {
    if (rr[i] >= 300 && rr[i] <= 2000) {
      nn.add(rr[i]);
      at.add(i);
      nnTs.add(ts[i]);
    }
  }
  final runs = <List<double>>[];
  var run = <double>[];
  var lastKept = -2;
  var lastTs = 0.0;
  for (var i = 0; i < nn.length; i++) {
    var keep = true;
    if (nn.length > radius) {
      final lo = math.max(0, i - radius);
      final hi = math.min(nn.length - 1, i + radius);
      final neighbors = <double>[];
      for (var j = lo; j <= hi; j++) {
        if (j != i) neighbors.add(nn[j]);
      }
      final med = neighbors.length < 2 ? null : median(neighbors);
      if (med != null && med > 0) keep = (nn[i] - med).abs() / med <= threshold;
    }
    if (!keep) {
      if (run.isNotEmpty) {
        runs.add(run);
        run = <double>[];
      }
      continue;
    }
    if (run.isNotEmpty &&
        (at[i] != lastKept + 1 || nnTs[i] - lastTs > nn[i] + 1000.0)) {
      runs.add(run);
      run = <double>[];
    }
    run.add(nn[i]);
    lastKept = at[i];
    lastTs = nnTs[i];
  }
  if (run.isNotEmpty) runs.add(run);
  return runs;
}
