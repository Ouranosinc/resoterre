cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  cwltool: "http://commonwl.org/cwltool#"


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

      - entry: $(inputs.logs_dir)
        entryname: tmp/logs
        writable: true

      - entry: $(inputs.figures_dir)
        entryname: tmp/figures
        writable: true

arguments:
  - $(inputs.config.path)

inputs:
  config:
    type: File
    doc: Inference configuration YAML

  inputs_dir:
    type: Directory
    doc: Input NetCDF files

  logs_dir:
    type: Directory
    doc: Logs directory

  figures_dir:
    type: Directory
    doc: Figures directory

outputs:
  outputs_dir:
    type: Directory
    outputBinding:
      glob: .
