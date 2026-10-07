cwlVersion: v1.2
class: Workflow
$namespaces:
  s: "https://schema.org/"
  iana: "https://www.iana.org/assignments/media-types/"
  edam: "http://edamontology.org/"

label: RDPS-to-HRDPS climate downscaling with a U-Net
doc: |
  Chain the RDPS forecast file listing with the UNet downscaling inference.
  The first step derives the RDPS files required for the requested datetime range,
  the second step downloads them into a local directory and the last step runs the
  inference on that directory.

s:dateCreated: "2026-09-09T10:36:53-04:00"
s:softwareVersion: "0.1.3-dev.10"
s:codeRepository: "https://github.com/Ouranosinc/resoterre"
s:license: "https://spdx.org/licenses/Apache-2.0"
s:citation: "https://github.com/Ouranosinc/resoterre/blob/0.1.3-dev.10/CITATION.cff"
s:releaseNotes: "https://github.com/Ouranosinc/resoterre/blob/0.1.3-dev.10/CHANGELOG.rst"
s:isBasedOn: "https://raw.githubusercontent.com/Ouranosinc/resoterre/0.1.3-dev.10/notebooks/ml-model-package/unet_rdps_to_hrdps/unet_rdps_to_hrdps.json"

s:author:
  - class: s:Person
    s:identifier: "https://orcid.org/0009-0004-9049-2092"
    s:email: gauvin-st-denis.blaise@ouranos.ca
    s:name: Blaise Gauvin St-Denis
    s:affiliation: Ouranos
  - class: s:Person
    s:identifier: "https://orcid.org/0000-0001-5393-8359"
    s:email: smith.trevorj@ouranos.ca
    s:name: Trevor James Smith
    s:affiliation: Ouranos
s:contributor:
  - class: s:Person
    s:identifier: "https://orcid.org/0000-0003-4862-3349"
    s:email: francis.charette-migneault@luqia.ca
    s:name: Francis Charette-Migneault
    s:affiliation: Technologies Luqia
  - class: s:Person
    s:email: nazim.azeli@luqia.ca
    s:name: Nazim Azeli
    s:affiliation: Technologies Luqia

s:keywords:
  - RDPS
  - HRDPS
  - downscaling
  - super-resolution
  - U-Net
  - machine learning
  - inference
  - climate
  - workflow

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
    format: iana:text/plain
    outputSource: downscaling_generate_file_list/forecast_files
  inference_output:
    type: Directory
    outputSource: downscaling_unet/inference_output

steps:
  downscaling_generate_file_list:
    run: downscaling_generate_file_list.cwl
    in:
      start_datetime: start_datetime
      end_datetime: end_datetime
      requires_previous_forecast_step: requires_previous_forecast_step
      include_year_month_subdirectory: include_year_month_subdirectory
      data_root: data_root
    out: [forecast_files]

  downscaling_download_files:
    doc: Download the listed RDPS files into a flat local directory.
    run: downscaling_download_files.cwl # when executing with weaver use  downscaling_download_files
    in:
      file_list: downscaling_generate_file_list/forecast_files
    out: [input_data]

  downscaling_unet:
    run: downscaling_unet.cwl
    in:
      config: config
      input_data: downscaling_download_files/input_data
      start_datetime: start_datetime
      end_datetime: end_datetime
    out: [inference_output]
