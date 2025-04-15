from gladier import GladierBaseTool


class InputDataTransfer(GladierBaseTool):

    flow_definition = {
        'Comment': 'Transfer the input data directory.',
        'StartAt': 'InputDataTransfer',
        'States': {
            'InputDataTransfer': {
                'Comment': 'Transfer the input data directory.',
                'Type': 'Action',
                'ActionUrl': 'https://actions.automate.globus.org/transfer/transfer',
                'Parameters': {
                    'source_endpoint_id.$': '$.input.local_endpoint_id',
                    'destination_endpoint_id.$': '$.input.remote_endpoint_id',
                    'transfer_items': [
                        {
                            'source_path.$': '$.input.input_source_path',
                            'destination_path.$': '$.input.input_destination_path',
                            'recursive.$': '$.input.input_transfer_recursive',
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
        'input_transfer_recursive': True,

    }
    required_input = [
        'input_source_path',
        'input_destination_path',
        'local_endpoint_id',
        'remote_endpoint_id',
    ]