import grpc
import pathlib

# import the generated classes
import service.service_spec.semantic_segmentation_aerial_pb2_grpc as grpc_bt_grpc
import service.service_spec.semantic_segmentation_aerial_pb2 as grpc_bt_pb2

from service import registry, base64_to_jpg, clear_file

if __name__ == "__main__":

    try:

        # open a gRPC channel
        endpoint = "localhost:{}".format(registry["semantic_segmentation_aerial_service"]["grpc"])
        channel = grpc.insecure_channel("{}".format(endpoint))
        print("Opened channel")

        grpc_method = "segment_aerial_image"

        input = pathlib.Path.cwd() / "docs/assets/examples/top_mosaic_09cm_area11.tif"
        input = str(input)
        print("Input file path: {}".format(input))
        window_size = 512
        stride = 128

        # create a stub (client)
        stub = grpc_bt_grpc.SemanticSegmentationAerialStub(channel)
        print("Stub created.")

        # create a valid request message
        request = grpc_bt_pb2.SemanticSegmentationAerialRequest(input=input,
                                                                window_size=window_size,
                                                                stride=stride)
        print("Request created.")
        # make the call
        response = stub.segment_aerial_image(request)
        print("Response received: {}".format(response))

        # et voilà
        output_file_path = "./segmented_output_image.jpg"
        if response.data:
            base64_to_jpg(response.data, output_file_path)
            clear_file(output_file_path)
            print("Service completed!")
        else:
            print("Service failed! No data received.")

    except Exception as e:
        print(e)
