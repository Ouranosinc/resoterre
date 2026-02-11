cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  ogc: "http://www.opengis.net/def/media-type/ogc/1.0/"

requirements:
  EnvVarRequirement:
    envDef:
      # to fix KeyError: 'getpwuid(): uid not found: 13798' in pytorch caching
      TORCHINDUCTOR_CACHE_DIR: "/tmp/torch_cache"
      HOME: "/tmp"
      USER: "cwluser"

  DockerRequirement:
    dockerPull: resoterre-inference:2026-01-26 # Change with image containing the model
    dockerOutputDirectory: /outputs

hints:
  cwltool:CUDARequirement:
    cudaComputeCapability: '3.0'
    cudaDeviceCountMax: 8
    cudaDeviceCountMin: 1
    cudaVersionMin: '11.4'


  InitialWorkDirRequirement:
    listing:
      - entry: $(inputs.inputs_dir)
        entryname: inputs
        writable: false

arguments:
  - $(inputs.config.path)

inputs:
  config:
    type: File
    doc: Inference configuration YAML

  inputs_dir:
    type: Directory
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
