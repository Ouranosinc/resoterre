cwlVersion: v1.2
class: Workflow
$namespaces:
  iana: "https://www.iana.org/assignments/media-types/"
  edam: "http://edamontology.org/"

doc: |
  Chain the RDPS forecast file listing with the UNet downscaling inference.
  The first step derives the RDPS files required for the requested datetime range,
  the second step downloads them into a local directory and the last step runs the
  inference on that directory.

inputs:
  start_datetime:
    type: string
    doc: Start datetime in ISO 8601 format, for example 2024-05-01T07:00:00.
  end_datetime:
    type: string
    doc: End datetime in ISO 8601 format, for example 2024-05-01T08:00:00.
  requires_previous_forecast_step:
    type: boolean
    default: true
    doc: |
      Whether the previous forecast step is required. Must be true for cumulative variables,
      such as precipitation.
  include_year_month_subdirectory:
    type: boolean
    default: true
    doc: |
      Whether the listed paths include the year/month subdirectory, as on thredds (true),
      or are stored in a flat directory (false).
  data_root:
    type: string
    default: "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/fileServer/birdhouse/disk3/ouranos/geoconnections/RDPS/"
    doc: Root location prepended to each listed file, used to download the RDPS files.
  config:
    type: ["null", File]
    format:
    - "iana:application/yaml"
    - "edam:format_3750"
    doc: Optional inference configuration YAML overriding the one built into the Docker image.

outputs:
  forecast_files:
    type: File
    format: http://edamontology.org/format_3475
    outputSource: generate_file_list/forecast_files
  inference_output:
    type: Directory
    outputSource: inference/inference_output

steps:
  generate_file_list:
    run: ../deploy/generate_file_list.cwl
    in:
      start_datetime: start_datetime
      end_datetime: end_datetime
      requires_previous_forecast_step: requires_previous_forecast_step
      include_year_month_subdirectory: include_year_month_subdirectory
      data_root: data_root
    out: [forecast_files]

  download_files:
    doc: Download the listed RDPS files into a flat local directory.
    run: ../deploy/download_files.cwl
    in:
      file_list: generate_file_list/forecast_files
    out: [input_data]

  inference:
    run: ../deploy/unet.cwl
    in:
      config: config
      input_data: download_files/input_data
    out: [inference_output]

