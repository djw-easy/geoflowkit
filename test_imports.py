from geoflow.flow import Flow
from geoflow.flowseries import FlowSeries
from geoflow.flowdataframe import FlowDataFrame






if __name__ == '__main__':
    # Create sample Flow objects
    flow1 = Flow([[0, 0], [1, 1]])
    flow2 = Flow([[1, 1], [2, 2]])
    flow3 = Flow([[2, 2], [3, 3]])

    # Create sample data
    data = {
        'id': [1, 2, 3],
        'value': [10, 20, 30],
        'geometry': FlowSeries([flow1, flow2, flow3])
    }
    fdf = FlowDataFrame(data, crs="EPSG:4326")


