"""Register sources command arguments."""

from pathlib import Path


def arguments(commands):
    open_images_index = commands.add_parser(
        "openimages-index", help="index relevant Open Images metadata"
    )
    open_images_index.add_argument(
        "--metadata-dir", type=Path, action="append", required=True
    )
    open_images_index.add_argument("--output-dir", type=Path, required=True)

    open_images_shortlist = commands.add_parser(
        "openimages-shortlist", help="create a deterministic download/review pool"
    )
    open_images_shortlist.add_argument("--candidates", type=Path, required=True)
    open_images_shortlist.add_argument("--output", type=Path, required=True)

    open_images_download = commands.add_parser(
        "openimages-download", help="download and validate a deterministic candidate pool"
    )
    open_images_download.add_argument("--shortlist", type=Path, required=True)
    open_images_download.add_argument("--image-dir", type=Path, required=True)
    open_images_download.add_argument("--manifest", type=Path, required=True)
    open_images_download.add_argument("--rejections", type=Path, required=True)
    open_images_download.add_argument("--compiled-class", action="append", default=[])
    open_images_download.add_argument("--candidate-kind", action="append", default=[])
    open_images_download.add_argument("--limit", type=int)
    open_images_download.add_argument("--workers", type=int, default=8)
    open_images_download.add_argument("--timeout", type=float, default=30)

    phenocam_index = commands.add_parser(
        "phenocam-index", help="index the approved PhenoCam v3 CMR collection"
    )
    phenocam_index.add_argument("--output-dir", type=Path, required=True)

    phenocam_sites = commands.add_parser(
        "phenocam-fetch-sites", help="retrieve authenticated PhenoCam site metadata"
    )
    phenocam_sites.add_argument("--index", type=Path, required=True)
    phenocam_sites.add_argument("--output", type=Path, required=True)
    phenocam_sites.add_argument("--netrc", type=Path)

    phenocam_plan = commands.add_parser(
        "phenocam-plan", help="select a bounded, seasonal PhenoCam archive pool"
    )
    phenocam_plan.add_argument("--granules", type=Path, required=True)
    phenocam_plan.add_argument("--sites", type=Path, required=True)
    phenocam_plan.add_argument("--output", type=Path, required=True)
    phenocam_plan.add_argument("--statistics", type=Path, required=True)

    phenocam_download = commands.add_parser(
        "phenocam-download", help="download and verify an approved archive plan"
    )
    phenocam_download.add_argument("--plan", type=Path, required=True)
    phenocam_download.add_argument("--archive-dir", type=Path, required=True)
    phenocam_download.add_argument("--manifest", type=Path, required=True)
    phenocam_download.add_argument("--rejections", type=Path, required=True)
    phenocam_download.add_argument("--netrc", type=Path)

    phenocam_sample = commands.add_parser(
        "phenocam-sample", help="safely sample visible frames from verified archives"
    )
    phenocam_sample.add_argument("--downloads", type=Path, required=True)
    phenocam_sample.add_argument("--frame-dir", type=Path, required=True)
    phenocam_sample.add_argument("--output", type=Path, required=True)
    phenocam_sample.add_argument("--rejections", type=Path, required=True)

    phenocam_select = commands.add_parser(
        "phenocam-select", help="make a conservative provisional 350/750 selection"
    )
    phenocam_select.add_argument("--screened", type=Path, required=True)
    phenocam_select.add_argument("--duplicate-pairs", type=Path, required=True)
    phenocam_select.add_argument("--output", type=Path, required=True)
    phenocam_select.add_argument("--statistics", type=Path, required=True)

    phenocam_review = commands.add_parser(
        "phenocam-review-packet", help="create mandatory PhenoCam annotation artifacts"
    )
    phenocam_review.add_argument("--selection", type=Path, required=True)
    phenocam_review.add_argument("--output-dir", type=Path, required=True)
