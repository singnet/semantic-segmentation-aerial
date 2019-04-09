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

        input_path = pathlib.Path.cwd() / "docs/assets/examples/top_mosaic_09cm_area11.tif"
        input_path = str(input_path)
        window_size = 512
        stride = 512

        # create a stub (client)
        stub = grpc_bt_grpc.SemanticSegmentationAerialStub(channel)
        print("Stub created.")

        # create a valid request message
        request = grpc_bt_pb2.SemanticSegmentationAerialRequest(input=input_path,
                                                                window_size=window_size,
                                                                stride=stride)
        print("Request created: {}".format(request))
        # make the call
        response = stub.segment_aerial_image(request)
        print("Response received! First 200 characters: {}".format(response.data[0:200]))
        if "out of memory" in response.data:
            exit(0)
        else:
            # et voilà
            output_file_path = "./segmented_output_image.jpg"
            if response.data:
                base64_to_jpg(response.data, output_file_path)
                clear_file(output_file_path)
                print("Service completed!")
            else:
                print("Service failed! No data received.")
            exit(0)
    except Exception as e:
        print(e)
        exit(1)
