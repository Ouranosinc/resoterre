cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  s: "https://schema.org/"
  iana: "https://www.iana.org/assignments/media-types/"
  edam: "http://edamontology.org/"

label: List RDPS forecast files
doc: |
  List the RDPS forecast files required to cover the requested datetime range.
  Each listed file is prefixed with the provided data root so the result can be
  consumed directly by the download step.

s:dateCreated: "2026-09-04T11:45:32-04:00"
s:version: "0.1.3.dev10"
s:softwareVersion: "0.1.3-dev.10"
s:codeRepository: "https://github.com/Ouranosinc/resoterre"
s:license: "https://spdx.org/licenses/Apache-2.0"
s:citation: "https://github.com/Ouranosinc/resoterre/blob/0.1.3-dev.10/CITATION.cff"
s:releaseNotes: "https://github.com/Ouranosinc/resoterre/blob/0.1.3-dev.10/CHANGELOG.rst"

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
  - forecast
  - file listing
  - downscaling

requirements:
  InlineJavascriptRequirement: {}
  DockerRequirement:
    dockerPull: resoterre-base:latest

baseCommand: [python3, -c]

arguments:
  - |
      from datetime import datetime
      from resoterre.experiments.rdps_to_hrdps_downscaling.rdps_to_hrdps_workflow import rdps_datetimes_to_forecast_files
      import sys
      start_datetime = datetime.fromisoformat(sys.argv[1])
      end_datetime = datetime.fromisoformat(sys.argv[2])
      files = rdps_datetimes_to_forecast_files(
          start_datetime,
          end_datetime,
          requires_previous_forecast_step=sys.argv[3] == "1",
          include_year_month_subdirectory=sys.argv[4] == "1",
      )
      data_root = sys.argv[5].rstrip("/")
      print("\n".join(f"{data_root}/{path}" if data_root else str(path) for path in files))

inputs:
  start_datetime:
    type: string
    doc: Start datetime in ISO 8601 format, for example 2024-05-01T07:00:00.
    inputBinding:
      position: 2
  end_datetime:
    type: string
    doc: End datetime in ISO 8601 format, for example 2024-05-01T08:00:00.
    inputBinding:
      position: 3
  requires_previous_forecast_step:
    type: boolean
    default: true
    doc: |
      Whether the previous forecast step is required. Must be true for cumulative variables,
      such as precipitation.
    inputBinding:
      position: 4
      valueFrom: '$(self ? "1" : "0")'
  include_year_month_subdirectory:
    type: boolean
    default: true
    doc: |
      Whether the file paths include the year/month subdirectory, as on thredds (true),
      or are stored in a flat directory (false).
    inputBinding:
      position: 5
      valueFrom: '$(self ? "1" : "0")'
  data_root:
    type: string
    default: "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/fileServer/birdhouse/disk3/ouranos/geoconnections/RDPS/"
    doc: |
      Root location prepended to each listed file. Set to an empty string to get relative paths.
    inputBinding:
      position: 6

outputs:
  forecast_files:
    type: File
    format: iana:text/plain
    outputBinding:
      glob: rdps_forecast_files.txt

stdout: rdps_forecast_files.txt
