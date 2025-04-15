from gladier import GladierBaseTool, generate_flow_definition


def remote_indexing(**data) -> int:
    import subprocess

    pathbins = '/eagle/projects/APSDataAnalysis/hparraga/laue-indexing'
    peaksearchPath = pathbins + '/peaksearch/peaksearch'
    p2qPath = pathbins + 'pixels2qs/pix2qs'
    indexingPath = pathbins + '/euler/euler'
    script = pathbins + '/pipeline/pyLaueGo.py'
    cmd = ['mpirun', '-np', '1', 'python', script,
        '--configFile', f"{data['remote_path']}{data['config_destination_path']}", # + \
        '--crystFile', f"{data['remote_path']}{data['crystal_destination_path']}",
        '--geoFile', f"{data['remote_path']}{data['geo_destination_path']}",
        '--filefolder', f"{data['remote_path']}{data['input_destination_path']}",
        '--outputFolder', f"{data['remote_path']}{data['output_source_path']}",
        '--peaksearchPath', peaksearchPath,
        '--p2qPath', p2qPath,
        '--indexingPath', indexingPath,
        '--filenamePrefix', f"{data['label']}_"]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode, result.stdout.decode("utf-8"), result.stderr.decode("utf-8")

@generate_flow_definition(modifiers={
    remote_indexing: {'WaitTime': 1000000,
                      'ExceptionOnActionFailure': True}
})
class RemoteIndexing(GladierBaseTool):

    required_input = [
        'funcx_endpoint_compute',
        'config_destination_path',
        'crystal_destination_path',
        'geo_destination_path',
        'input_destination_path',
        'output_source_path',
        'remote_path',
        'label'
    ]

    funcx_functions = [
        remote_indexing
    ]