from gladier import GladierBaseTool


class OutputDataTransfer(GladierBaseTool):

    flow_definition = {
        'Comment': 'Transfer the results directory.',
        'StartAt': 'OutputDataTransfer',
        'States': {
            'OutputDataTransfer': {
                'Comment': 'Transfer the results directory.',
                'Type': 'Action',
                'ActionUrl': 'https://actions.automate.globus.org/transfer/transfer',
                'Parameters': {
                    'source_endpoint_id.$': '$.input.remote_endpoint_id',
                    'destination_endpoint_id.$': '$.input.local_endpoint_id',
                    'transfer_items': [
                        {
                            'source_path.$': '$.input.output_source_path',
                            'destination_path.$': '$.input.output_destination_path',
                            'recursive.$': '$.input.output_transfer_recursive',
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
        'output_transfer_recursive': True,
    }
    required_input = [
        'output_source_path',
        'output_destination_path',
        'local_endpoint_id',
        'remote_endpoint_id',
    ]