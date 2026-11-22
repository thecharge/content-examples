# Source and alignment audit

The experiment reuses the measured benchmark's UCI, Danish apartment and ZEB input checksum checks, row membership and split definitions. UCI's training release is split after raw row 6,499. Test1 and test2 stay separate. Danish apartments are an external comparison using operative temperature, relative humidity and CO₂ with logbook labels. Heating uses ZEB Living Lab member `ds2-detailed-PT1M.csv`, a chronological pre-March-7 fit/validation split and a March-7-and-later logged-action test. These three sources have been explored in earlier measurements. The benchmark records publisher archives and licenses where given. [Measured benchmark](../attention-and-reward/README.md)

The fresh occupancy source is Christoph Siegl, Thomas Hirsch, Theresa Kohl, Franz Wotawa and Gerald Schweiger, *Data and Sourcecode from: Neural Network-based Occupancy Detection on the Edge*, version 2, published June 11, 2024. The Zenodo API identifies the record license as AGPL-3.0-or-later and lists MD5 `58c8fc3cac8723c7fa1a7e1e61e86a40` for the archive. The local file matched that MD5 and the supplied SHA-256 in the freeze. The record describes an original sensor dump at five-minute intervals and manually labeled image counts. It identifies the separate `occupancy` field in Room A features as an IR sensor. We excluded that field, every other occupancy proxy field, all motion and event sensors, and image counts from feature construction. [Zenodo record](https://zenodo.org/records/11573783) · [Record metadata](https://zenodo.org/api/records/11573783)

## Raw-to-UTC alignment

Room A begins at raw sensor time 2023-09-12 09:05 UTC with raw image time 11:05 local. Room B begins at raw sensor time 2023-11-16 11:00 UTC with raw image time 12:00 local. Europe/Vienna conversion accounts for both offsets. Every overlapping published intermediate image count agreed with the raw image count at the joined UTC timestamp. The project did not use the intermediate filtered population for evaluation. It uses all valid raw images with complete strictly earlier feature history. No missing labels were imputed.

| Count | Room A | Room B |
| --- | ---: | ---: |
| Raw sensor rows | 28,548 | 9,792 |
| Raw image rows | 28,536 | 9,792 |
| Raw sensor timestamps without a raw label | 12 | 0 |
| Ambiguous local image timestamps excluded | 12 | 0 |
| Unambiguous label timestamps missing an exact sensor timestamp | 0 | 0 |
| Missing or invalid preceding history | 9 | 9 |
| Eligible label windows | 28,515 | 9,783 |
| Occupied windows | 6,917 | 2,119 |
| Intermediate image counts that match raw at UTC time | 23,855 | 6,451 |
| Intermediate image counts that differ at UTC time | 0 | 0 |

A separate raw-to-intermediate feature comparison found exact temperature/humidity/CO₂ triplets for all 23,855 Room A intermediate timestamps and 6,451 Room B timestamps, with zero changed triplets or absent raw timestamps. No experiment feature was averaged, resampled, forward-filled or joined to a future row. The record describes five-minute acquisition but does not specify the sensor firmware's averaging interval. The design therefore drops the sensor rows exactly at the label and five minutes before it. The latest used timestamp lies ten minutes before the label, giving a full five-minute separation even if timestamps name the starts of acquisition bins. This audit cannot establish details of any undocumented sensor firmware buffering.

The delta recurrence uses the independently controlled forgetting dimensions described in the Kimi Linear paper, particularly §3 on Kimi Delta Attention. Its small single-head implementation has learned write and output gates and uses a sequential delta residual update. This experiment does not benchmark Kimi Linear's chunkwise kernel, large-language-model quality, or throughput. The paper page and appendix-linked §3 heading were checked November 22, 2026. [Kimi Team, Kimi Linear](https://arxiv.org/html/2510.26692#S3)
