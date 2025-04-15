#!/usr/bin/env python

import argparse

##Base Gladier imports
from gladier import GladierBaseClient, generate_flow_definition

##Import tools that will be used on the flow definition
from tools.configTransfer import ConfigTransfer
from tools.crystalTransfer import CrystalTransfer
from tools.geoTransfer import GeoTransfer
from tools.inputDataTransfer import InputDataTransfer
from tools.remoteIndexing import RemoteIndexing
from tools.outputDataTransfer import OutputDataTransfer

##Generate flow based on the collection of gladier tools
@generate_flow_definition
class LaueIndexing(GladierBaseClient):
    gladier_tools = [
        ConfigTransfer,
        CrystalTransfer,
        GeoTransfer,
        InputDataTransfer,
        RemoteIndexing,
        OutputDataTransfer,
    ]

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label', required=True, help='Label for this run')
    parser.add_argument('--configFile', required=True, help='Path to the config file')
    parser.add_argument('--crystalFile', required=True, help='Path to the crystal structure file')
    parser.add_argument('--geoFile', required=True, help='Path to the geometry file')
    parser.add_argument('--inputDir', required=True, help='Path to the input directory')
    parser.add_argument('--outputDir', required=True, help='Path to the output directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    #aps#clutch
    localEndpoint = 'b0e921df-6d04-11e5-ba46-22000b92c6ec'
    #eagle guest collection
    remoteEndpoint = 'fd5a47b6-5fb0-4d85-b5eb-81ae993b1291'
    remotePath = '/eagle/APSDataProcessing/aps34ide'
    #globus compute personal endpoint
    computeEndpoint = '347c30d8-4adc-4522-8774-b451a60c0923'
    #staging directory on eagle
    stagingDir = f'/staging/{args.label}'
    configDir = f'{stagingDir}/config'
    remoteInputDir = f'{stagingDir}/data'
    remoteOutputDir = stagingDir #f'{stagingDir}/processed'

    flow_input = {
        'input': {
            #endpoints
            'local_endpoint_id': localEndpoint,
            'remote_endpoint_id': remoteEndpoint,
            'funcx_endpoint_compute': computeEndpoint,

            #transfer files
            'config_source_path': args.configFile,
            'config_destination_path': f'{configDir}/config.yml',
            'crystal_source_path': args.crystalFile,
            'crystal_destination_path': f'{configDir}/crystal.xml',
            'geo_source_path': args.geoFile,
            'geo_destination_path': f'{configDir}/geo.xml',
            'input_source_path': args.inputDir,
            'input_destination_path': remoteInputDir,
            'output_source_path': remoteOutputDir,
            'output_destination_path': args.outputDir,

            'label': args.label,
            'remote_path': remotePath
        }
    }

    client = LaueIndexing()

    #Flow execution
    flowRun = client.run_flow(flow_input=flow_input, label=f'Laue Indexing {args.label}')

    print(f"Run started with ID: {flowRun['action_id']}")
    print(f"https://app.globus.org/runs/{flowRun['action_id']}")