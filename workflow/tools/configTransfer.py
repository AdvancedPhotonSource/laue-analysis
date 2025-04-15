from gladier import GladierBaseTool


class ConfigTransfer(GladierBaseTool):

    flow_definition = {
        'Comment': 'Transfer the config structure file.',
        'StartAt': 'ConfigTransfer',
        'States': {
            'ConfigTransfer': {
                'Comment': 'Transfer the config structure file.',
                'Type': 'Action',
                'ActionUrl': 'https://actions.automate.globus.org/transfer/transfer',
                'Parameters': {
                    'source_endpoint_id.$': '$.input.local_endpoint_id',
                    'destination_endpoint_id.$': '$.input.remote_endpoint_id',
                    'transfer_items': [
                        {
                            'source_path.$': '$.input.config_source_path',
                            'destination_path.$': '$.input.config_destination_path',
                            'recursive.$': '$.input.config_transfer_recursive',
                        }
                    ]
                },
                'ResultPath': '$.Transfer',
                'WaitTime': 600,
                'End': True
            },
        }
    }

    flow_input = {
        'transfer_sync_level': 'checksum',
        'config_transfer_recursive': False,

    }
    required_input = [
        'config_source_path',
        'config_destination_path',
        'local_endpoint_id',
        'remote_endpoint_id',
    ]