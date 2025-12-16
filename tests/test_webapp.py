from rdk_generator.webapp.app import create_app


def test_index_loads():
    app = create_app()
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"RDK Generator" in r.data
