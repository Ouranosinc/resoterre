cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  s: "https://schema.org/"
  cwltool: "http://commonwl.org/cwltool#"
  iana: "https://www.iana.org/assignments/media-types/"
  edam: "http://edamontology.org/"

label: RDPS-to-HRDPS U-Net downscaling inference
doc: |
  Run the RDPS to HRDPS U-Net downscaling inference over the provided input directory.
  The model, regridding matrices, geophysical fields and default configuration are baked
  into the inference image.

s:dateCreated: "2026-01-30T10:42:53-05:00"
s:version: "0.1.3.dev10"
s:softwareVersion: "v0.1.3-dev.10"
s:codeRepository: "https://github.com/Ouranosinc/resoterre"
s:license: "https://spdx.org/licenses/Apache-2.0"
s:citation: "https://github.com/Ouranosinc/resoterre/blob/v0.1.3-dev.10/CITATION.cff"
s:releaseNotes: "https://github.com/Ouranosinc/resoterre/blob/v0.1.3-dev.10/CHANGELOG.rst"
s:isBasedOn: "https://raw.githubusercontent.com/Ouranosinc/resoterre/v0.1.3-dev.10/notebooks/ml-model-package/unet_rdps_to_hrdps/unet_rdps_to_hrdps.json"

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

requirements:
  InlineJavascriptRequirement: {}

  EnvVarRequirement:
    envDef:
      # to fix KeyError: 'getpwuid(): uid not found: 13798' in pytorch caching
      TORCHINDUCTOR_CACHE_DIR: "/tmp/torch_cache"
      HOME: "/tmp"
      USER: "cwluser"

  DockerRequirement:
    dockerPull: resoterre-inference:latest # Change with image containing the model

  InitialWorkDirRequirement:
    listing:
      - entry: $(inputs.config)
        entryname: config.yaml
        writable: false
      - entry: $(inputs.input_data)
        entryname: inputs
        writable: false

hints:
  cwltool:CUDARequirement:
    cudaComputeCapability: '3.0'
    cudaDeviceCountMax: 8
    cudaDeviceCountMin: 1
    cudaVersionMin: '11.4'

baseCommand: []

inputs:
  config:
    type: ["null", File]
    format:
    - "iana:application/yaml"
    - "edam:format_3750"
    doc: Inference configuration YAML
    inputBinding:
      position: 1
      prefix: --config
      valueFrom: config_yaml=config.yaml

  start_datetime:
    type: ["null", string]
    doc: Optional start datetime (ISO 8601) overriding the config's inference_start_datetime
    inputBinding:
      position: 2
      valueFrom: '$(self ? "start_datetime=" + self : null)'

  end_datetime:
    type: ["null", string]
    doc: Optional end datetime (ISO 8601) overriding the config's inference_end_datetime
    inputBinding:
      position: 3
      valueFrom: '$(self ? "end_datetime=" + self : null)'

  input_data:
    type: Directory
    doc: Directory containing input NetCDF files to be used for inference

  number_of_cores:
    type: boolean
    default: true
    doc: Internal flag; always emits -j1 to set the number of cores used for inference.
    inputBinding:
      position: 4
      valueFrom: "-j1"

  output_directory:
    type: boolean
    default: true
    doc: Internal flag; injects the runtime-determined output directory into the command line.
    inputBinding:
      position: 5
      valueFrom: '$("--directory=" + runtime.outdir)'

outputs:
  inference_output:
    type: Directory
    doc: Zarr output directory containing inference results
    outputBinding:
      glob: outputs/inference_*.zarr
