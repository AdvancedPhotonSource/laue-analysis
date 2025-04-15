from gladier import GladierBaseTool


class CrystalTransfer(GladierBaseTool):

    flow_definition = {
        'Comment': 'Transfer the crystal structure file.',
        'StartAt': 'CrystalTransfer',
        'States': {
            'CrystalTransfer': {
                'Comment': 'Transfer the crystal structure file.',
                'Type': 'Action',
                'ActionUrl': 'https://actions.automate.globus.org/transfer/transfer',
                'Parameters': {
                    'source_endpoint_id.$': '$.input.local_endpoint_id',
                    'destination_endpoint_id.$': '$.input.remote_endpoint_id',
                    'transfer_items': [
                        {
                            'source_path.$': '$.input.crystal_source_path',
                            'destination_path.$': '$.input.crystal_destination_path',
                            'recursive.$': '$.input.crystal_transfer_recursive',
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
        'crystal_transfer_recursive': False,

    }
    required_input = [
        'crystal_source_path',
        'crystal_destination_path',
        'local_endpoint_id',
        'remote_endpoint_id',
    ]