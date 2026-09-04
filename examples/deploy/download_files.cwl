cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  iana: "https://www.iana.org/assignments/media-types/"
  edam: "http://edamontology.org/"

doc: |
  Download every file referenced in a file list into a flat directory named inputs.
  Entries can be HTTP(S) URLs or local paths reachable from the container.

requirements:
  NetworkAccess:
    networkAccess: true
  DockerRequirement:
    dockerPull: resoterre-base:latest

baseCommand: [python3, -c]

arguments:
  - position: 1
    valueFrom: |
      import shutil
      import sys
      import urllib.request
      from pathlib import Path
      destination = Path("inputs")
      destination.mkdir(exist_ok=True)
      references = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
      for reference in references:
          target = destination / reference.rsplit("/", 1)[-1]
          if "://" in reference:
              with urllib.request.urlopen(reference) as response, target.open("wb") as handle:  # noqa: S310
                  shutil.copyfileobj(response, handle)
          else:
              shutil.copyfile(reference, target)
          print(f"{reference} -> {target}", file=sys.stderr)

inputs:
  file_list:
    type: File
    format: http://edamontology.org/format_3475
    doc: Text file holding one file reference per line.
    inputBinding:
      position: 2

outputs:
  input_data:
    type: Directory
    doc: Directory holding the downloaded files, ready to be used as the UNet input_data.
    outputBinding:
      glob: inputs
