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
  # The runner picks the working directory at runtime, so it cannot be hardcoded in the image.
  - --directory=$(runtime.outdir)

inputs:
  config:
    type: ["null", File]
    format:
    - "iana:application/yaml"
    - "edam:format_3750"
    doc: Inference configuration YAML
    inputBinding:
      prefix: --config
      valueFrom: config_yaml=config.yaml

  input_data:
    type: Directory
    doc: Directory containing input NetCDF files to be used for inference

outputs:
  inference_output:
    type: Directory
    doc: Zarr output directory containing inference results
    outputBinding:
      glob: outputs/inference_*.zarr
