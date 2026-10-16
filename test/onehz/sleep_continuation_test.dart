import 'dart:math' as math;

import 'package:openstrap_analytics/onehz.dart';
import 'package:test/test.dart';

// Synthetic night; no personal sensor data. A return to sleep after an hour
// awake must survive the final main-sleep selection, with the gap still awake.
void main() {
  for (final scenario in [
    (gapMinutes: 70, observed: true),
    (gapMinutes: 100, observed: true),
    (gapMinutes: 70, observed: false),
  ]) {
    final gapMinutes = scenario.gapMinutes;
    test(
        'main sleep after a $gapMinutes-minute break (observed: ${scenario.observed})',
        () {
      final accel = <AccelSample>[];
      final hr = <double>[];
      final start = DateTime.utc(2024, 1, 1, 21).millisecondsSinceEpoch;
      var elapsedSeconds = 0;
      void block(int minutes, {required bool awake, bool observed = true}) {
        for (var i = 0; i < minutes * 60; i++, elapsedSeconds++) {
          if (!observed) continue;
          final phase = math.sin(i * 0.5);
          accel.add(AccelSample(
            start + elapsedSeconds * 1000.0,
            awake ? 0.3 * phase : 0.02,
            awake ? 0.3 : 0.02,
            awake ? 0.9 * (1 - 0.2 * phase) : 1.0,
          ));
          hr.add(awake ? 85 : 49 + 1.5 * math.sin(i / 1800));
        }
      }

      block(120, awake: true);
      block(240, awake: false); // 23:00–03:00
      block(gapMinutes, awake: true, observed: scenario.observed);
      block(180, awake: false);
      block(120, awake: true);

      final sleep = segmentSleep(accel, hr, tzOffsetSec: 0);
      expect(sleep.present, isTrue);
      if (gapMinutes == 70) {
        expect(sleep.inBedSec, greaterThan(8 * 3600),
            reason: 'both overnight blocks belong to the main sleep');
        expect(sleep.tstSec, greaterThan(6 * 3600));
        expect(sleep.tstSec, lessThanOrEqualTo(7 * 3600),
            reason: 'the awake gap must not inflate time asleep');
        if (scenario.observed) {
          expect(sleep.wasoSec, greaterThanOrEqualTo(65 * 60));
        } else {
          expect(sleep.unobservedSec, greaterThanOrEqualTo(70 * 60));
          expect(sleep.wasoSec, lessThan(10 * 60),
              reason: 'missing measurements must not become measured wake');
        }
      } else {
        expect(sleep.inBedSec, lessThan(5 * 3600),
            reason: 'a break beyond the continuation limit stays separate');
      }
    });
  }
}
