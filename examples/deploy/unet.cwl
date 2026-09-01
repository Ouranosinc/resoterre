cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  iana: "https://www.iana.org/assignments/media-types/"
  edam: "http://edamontology.org/"

requirements:
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

arguments:
  - -j1 # Define the number of cores to use for inference (1 core in this case)
  - --config
  - config_yaml=config.yaml

inputs:
  config:
    type: File
    format:
    - "iana:application/yaml"
    - "edam:format_3750"
    doc: Inference configuration YAML

  input_data:
    type: Directory
    doc: Directory containing input NetCDF files with folder structure (e.g., 202405/2024050100_007.nc)

outputs:
  inference_output:
    type: Directory
    doc: Zarr output directory containing inference results
    outputBinding:
      glob: outputs/inference_cwl.zarr
