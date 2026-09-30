"""Generated gRPC stubs (voiceguard_pb2*.py) -- do not hand-edit.

Regenerate after changing voiceguard.proto:
    python -m grpc_tools.protoc -I src/voiceguard/serve/proto \
        --python_out=src/voiceguard/serve/proto \
        --grpc_python_out=src/voiceguard/serve/proto \
        --pyi_out=src/voiceguard/serve/proto \
        src/voiceguard/serve/proto/voiceguard.proto

then re-apply the single import fixup noted at the top of
voiceguard_pb2_grpc.py (grpc_tools emits a bare `import voiceguard_pb2`,
which only resolves as a package-relative import here).
"""
