from mangum import Mangum

from openai_api.openai_api_app import app

mangum_handler = Mangum(app)


def openai_api_handler(event, context):
    return mangum_handler(event, context)
