cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  ogc: "http://www.opengis.net/def/media-type/ogc/1.0/"
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
    dockerPull: resoterre-inference:2026-01-26 # Change with image containing the model

hints:
  cwltool:CUDARequirement:
    cudaComputeCapability: '3.0'
    cudaDeviceCountMax: 8
    cudaDeviceCountMin: 1
    cudaVersionMin: '11.4'


  InitialWorkDirRequirement:
    listing:
      - entry: $(inputs.input)
        entryname: inputs/$(inputs.input.basename)
        writable: false

arguments:
  - $(inputs.config.path)
  - --preprocess_batch
  - inputs/$(inputs.input.basename)
  - --path_models
  - /app/model
  - --path_output
  - outputs

inputs:
  config:
    type: File
    format:
    - "iana:application/yaml"
    - "edam:format_3750"
    doc: Inference configuration YAML

  input:
    type: File
    format:
    - "ogc:netcdf"
    - "iana:application/netcdf"
    doc: Input NetCDF files

outputs:
  HRDPS_P_PR_SFC:
    type: File[]
    format: "ogc:netcdf"
    outputBinding:
      glob: outputs/HRDPS_P_PR_SFC/*.nc
  HRDPS_P_TT_10000:
    type: File[]
    format: "ogc:netcdf"
    outputBinding:
      glob: outputs/HRDPS_P_TT_10000/*.nc
  HRDPS_P_UUC_10000:
    type: File[]
    format: "ogc:netcdf"
    outputBinding:
      glob: outputs/HRDPS_P_UUC_10000/*.nc
  HRDPS_P_VVC_10000:
    type: File[]
    format: "ogc:netcdf"
    outputBinding:
      glob: outputs/HRDPS_P_VVC_10000/*.nc
