Version 0.2.14 `Jun 28, 2021`
-------------------------------

- Add parameter ``uns_white_list`` in ``filter_data`` function to keep QC statistics if needed.

Version 0.2.13 `Jun 24, 2021`
-------------------------------

- The ``aggregate_matrices`` function now accepts sample sheet in Python dictionary format besides a CSV file path string. See details in its description in API panel.

Version 0.2.12 `May 28, 2021`
-------------------------------

- Bug fix.

Version 0.2.11 `May 17, 2021`
-------------------------------

- Bug fix.

Version 0.2.10 `February 2, 2021`
----------------------------------

- Feature enhancement.

Version 0.2.9 `December 25, 2020`
-----------------------------------

- Fix a bug for caching percent mito rate.
- Improve `write_mtx` function.

Version 0.2.8 `December 7, 2020`
-----------------------------------

- Add support on loading ``loom`` file with Seurat-style cell barcode and feature key names.
- Bug fix: resolve an issue on count matrix dimension inconsistency with feature metadata on data aggregation, when last feature has ``0`` count across all cell barcodes. Thanks to `Mikhail Alperovich <misha.alperovich1@gmail.com>`_ for reporting this issue.
- Other bug fix and performance improvements.

Version 0.2.7 `October 13, 2020`
-----------------------------------

- Add support for Nanostring GeoMx data format.
- Fix bugs.

Version 0.2.6 `September 28, 2020`
-----------------------------------

Fix bug in SCP compatible output generation.

Version 0.2.5 `August 19, 2020`
--------------------------------
Adjustment for Pegasus command usage.

Version 0.2.2 `June 16th, 2020`
--------------------------------
Fix bugs in data aggregation.

Version 0.2.1 `June 8th, 2020`
--------------------------------
Fix bug in processing single ``h5`` file.

Version 0.2.0 `June 7th, 2020`
--------------------------------
Initial release.
