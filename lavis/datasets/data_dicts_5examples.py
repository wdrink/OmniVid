DATA_DICTS = {
    "activitynet-train": [
        {
            "total_chunks": 1976,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/activitynet/train/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/activitynet/train/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        }
    ],
    "activitynet-val": [
        {
            "total_chunks": 969,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/activitynet/val_1/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/activitynet/val_1/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
        {
            "total_chunks": 963,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/activitynet/val_2/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/activitynet/val_2/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "charades_mr-train": [
        {
            "total_chunks": 2481,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/charades_mr/train/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/charades_mr/train/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        }
    ],
    "charades_mr-test": [
        {
            "total_chunks": 744,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/charades_mr/test/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/charades_mr/test/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        }
    ],
    "k700-2020-train": [
        {
            "total_chunks": 107299,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/k700-2020/train/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/k700-2020/train/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "k700-2020-val": [
        {
            "total_chunks": 6793,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/k700-2020/val/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/k700-2020/val/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "thumos14-val": [
        {
            "total_chunks": 40,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/thumos14/val/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/thumos14/val/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "thumos14-test": [
        {
            "total_chunks": 42,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/thumos14/test/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/thumos14/test/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "MSR-VTT-train": [
        {
            "total_chunks": 1302,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/MSR-VTT/train/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/MSR-VTT/train/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "MSR-VTT-val": [
        {
            "total_chunks": 99,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/MSR-VTT/val/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/MSR-VTT/val/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "MSR-VTT-test": [
        {
            "total_chunks": 598,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/MSR-VTT/test/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/MSR-VTT/test/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "activitynet-tal-train": [
        {
            "total_chunks": 1979,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/activitynet_tal/train/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/activitynet_tal/train/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
    "activitynet-tal-val": [
        {
            "total_chunks": 971,
            "storage_account_url": "https://interndatasetsv2.blob.core.windows.net/",
            "container_name": "omnicaption",
            "SAS": "?sv=2020-10-02&st=2023-05-04T10%3A45%3A13Z&se=2023-09-05T10%3A45%3A00Z&sr=c&sp=racwdxlt&sig=5TN5u29kF1O%2BCA%2FOKn3nTr%2BQrrsMyg3p9JJY5swRfkk%3D",
            "image_blob_template": "binary_chunk_5examples/activitynet_tal/val/video/video-{:010d}",
            "text_blob_template": "binary_chunk_5examples/activitynet_tal/val/text/text-{:010d}.tsv",
            "image_data_type": "binary",
            "text_data_type": "tsv",
            "with_image_key": True,
        },
    ],
}