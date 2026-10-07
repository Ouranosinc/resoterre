cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  s: "https://schema.org/"
  iana: "https://www.iana.org/assignments/media-types/"
  edam: "http://edamontology.org/"

label: Download listed files
doc: |
  Download every file referenced in a file list into the output working directory.
  Entries can be HTTP(S) URLs or local paths reachable from the container.

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
  - download
  - downscaling

requirements:
  NetworkAccess:
    networkAccess: true
  DockerRequirement:
    dockerPull: resoterre-base:latest

baseCommand: [python3, -c]

arguments:
  - |
      import shutil
      import sys
      import urllib.request
      from pathlib import Path
      from urllib.parse import urlparse
      destination = Path(".")
      references = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
      for reference in references:
          target = destination / reference.rsplit("/", 1)[-1]
          scheme = urlparse(reference).scheme
          if scheme in ("http", "https"):
              with urllib.request.urlopen(reference) as response, target.open("wb") as handle:
                  shutil.copyfileobj(response, handle)
          else:
              source = reference[len("file://"):] if scheme == "file" else reference
              shutil.copyfile(source, target)
          print(f"{reference} -> {target}", file=sys.stderr)

inputs:
  file_list:
    type: File
    format: iana:text/plain
    doc: Text file holding one file reference per line.
    inputBinding:
      position: 2

outputs:
  input_data:
    type: Directory
    doc: Directory holding the downloaded files, ready to be used as the UNet input_data.
    outputBinding:
      glob: "."
