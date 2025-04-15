from gladier import GladierBaseTool


class GeoTransfer(GladierBaseTool):

    flow_definition = {
        'Comment': 'Transfer the geometry file.',
        'StartAt': 'GeoTransfer',
        'States': {
            'GeoTransfer': {
                'Comment': 'Transfer the geometry file.',
                'Type': 'Action',
                'ActionUrl': 'https://actions.automate.globus.org/transfer/transfer',
                'Parameters': {
                    'source_endpoint_id.$': '$.input.local_endpoint_id',
                    'destination_endpoint_id.$': '$.input.remote_endpoint_id',
                    'transfer_items': [
                        {
                            'source_path.$': '$.input.geo_source_path',
                            'destination_path.$': '$.input.geo_destination_path',
                            'recursive.$': '$.input.geo_transfer_recursive',
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
        'geo_transfer_recursive': False,

    }
    required_input = [
        'geo_source_path',
        'geo_destination_path',
        'local_endpoint_id',
        'remote_endpoint_id',
    ]