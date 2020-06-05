Use PegasusIO as a command line tool
======================================

PegasusIO can be used as a command line tool. Type::

    pegasus -h

to see the help information::

    Usage:
      pegasus <command> [<args>...]
      pegasus -h | --help
      pegasus -v | --version

PegasusIO currently has only one sub-command:

- ``aggregate_matrix``: Aggregate sample count matrices into a single count matrix. It also enables users to import metadata into the count matrix.

``pegasusio aggregate_matrix``
-------------------------------

pegasus aggregate_matrix allows aggregating arbitrary matrices with the help of a CSV format sample sheet.

Type::

    pegasusio aggregate_matrix -h

to see the usage information::

    Usage:
      pegasusio aggregate_matrix <csv_file> <output_name> [--restriction <restriction>... options]
      pegasusio aggregate_matrix -h

* Arguments:

    csv_file
        Input csv-formatted file containing information of each sc/snRNA-seq sample. This file must contain at least 2 columns - Sample, sample name and Location, location of the sample count matrix in either 10x v2/v3, DGE, mtx, csv, tsv or loom format. Additionally, an optional Reference column can be used to select samples generated from a same reference (e.g. mm10). If the count matrix is in either DGE, mtx, csv, tsv, or loom format, the value in this column will be used as the reference since the count matrix file does not contain reference name information. In addition, the Reference column can be used to aggregate count matrices generated from different genome versions or gene annotations together under a unified reference. For example, if we have one matrix generated from mm9 and the other one generated from mm10, we can write mm9_10 for these two matrices in their Reference column. Pegasus will change their references to 'mm9_10' and use the union of gene symbols from the two matrices as the gene symbols of the aggregated matrix. For HDF5 files (e.g. 10x v2/v3), the reference name contained in the file does not need to match the value in this column. In fact, we use this column to rename references in HDF5 files. For example, if we have two HDF files, one generated from mm9 and the other generated from mm10. We can set these two files' Reference column value to 'mm9_10', which will rename their reference names into mm9_10 and the aggregated matrix will contain all genes from either mm9 or mm10. This renaming feature does not work if one HDF5 file contain multiple references (e.g. mm10 and GRCh38). See below for an example csv::

            Sample,Source,Platform,Donor,Reference,Location
            sample_1,bone_marrow,NextSeq,1,GRCh38,/my_dir/sample_1/filtered_gene_bc_matrices_h5.h5
            sample_2,bone_marrow,NextSeq,2,GRCh38,/my_dir/sample_2/filtered_gene_bc_matrices_h5.h5
            sample_3,pbmc,NextSeq,1,GRCh38,/my_dir/sample_3/filtered_gene_bc_matrices_h5.h5
            sample_4,pbmc,NextSeq,2,GRCh38,/my_dir/sample_4/filtered_gene_bc_matrices_h5.h5

    output_name
        The output file name.

* Options:

    -\\-restriction <restriction>...
        Select data that satisfy all restrictions. Each restriction takes the format of name:value,...,value or name:~value,..,value, where ~ refers to not. You can specifiy multiple restrictions by setting this option multiple times.

    -\\-attributes <attributes>
        Specify a comma-separated list of outputted attributes. These attributes should be column names in the csv file.

    -\\-default-reference <reference>
        If sample count matrix is in either DGE, mtx, csv, tsv or loom format and there is no Reference column in the csv_file, use <reference> as the reference.

    -\\-select-only-singlets
        If we have demultiplexed data, turning on this option will make pegasusio only include barcodes that are predicted as singlets.

    -\\-min-genes <number>
         Only keep cells with at least <number> of genes.

    -\\-max-genes <number>
        Only keep cells with less than <number> of genes.

    -\\-min-umis <number>
        Only keep cells with at least <number> of UMIs.

    -\\-max-umis <number>
        Only keep cells with less than <number> of UMIs.

    -\\-mito-prefix <prefix>
        Prefix for mitochondrial genes. If multiple prefixes are provided, separate them by comma (e.g. "MT-,mt-").

    -\\-percent-mito <percent>
        Only keep cells with mitochondrial percent less than <percent>%. Only when both mito_prefix and percent_mito set, the mitochondrial filter will be triggered.

    -\\-no-append-sample-name
        Turn this option on if you do not want to append sample name in front of each sample's barcode (concatenated using '-').

    \-h, -\\-help
        Print out help information.

* Outputs:

    output_name.zarr.zip
        A zipped Zarr file containing aggregated data.

* Examples::

    pegasusio aggregate_matrix --restriction Source:BM,CB --restriction Individual:1-8 --attributes Source,Platform count_matrix.csv aggr_data
