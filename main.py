"""Cubist ML Abstraction Api.

Provides a simple HTTP api that allows Cubist engineers to
interact with one or many Machine Learning as a Service providers.
"""

__author__ = 'reinaldo@reinaldo.ca'

import binascii
import csv
import httplib2
import json
import logging
import StringIO

import lib.cloudstorage as gcs
from oauth2client.appengine import AppAssertionCredentials
from apiclient.discovery import build

import constants
import webapp2

http = AppAssertionCredentials(
    'https://www.googleapis.com/auth/prediction '
    'https://www.googleapis.com/auth/devstorage.full_control').authorize(
        httplib2.Http())
service = build('prediction', 'v1.6', http=http)


class MlWebserviceBase(webapp2.RequestHandler):
  """Base class used by all handlers in the Api."""

  def get(self):
    self.post()


class TrainHandler(MlWebserviceBase):
  """Sends training data to the backend and schedules
  a model for training. Sends to the browser the future
  id of the model. Currently uses Google Prediction Api
  as main backend.
  """

  def post(self):
    payload = self.request.get('data').encode(constants.UTF8_ENCODING_LABEL)
    hash = binascii.crc32(payload) & 0xffffffff
    write_retry_params = gcs.RetryParams(backoff_factor=1.1)
    filename = constants.CLOUD_FILE_PATTERN % (constants.CLOUD_STORAGE_BUCKET,
                                               hash)
    gcs_file = gcs.open(filename,
                        'w',
                        content_type=constants.OUTPUT_CONTENT_TYPE_TEXT,
                        retry_params=write_retry_params)
    gcs_file.write(payload)
    gcs_file.close()

    # Now let's train the model!
    result = service.trainedmodels().insert(
        project=constants.PROJECT_ID,
        body={
            'id': str(hash),
            'storageDataLocation': filename[1:],
            'modelType': 'REGRESSION',
        }).execute()

    self.response.headers['Content-Type'] = constants.OUTPUT_CONTENT_TYPE_TEXT
    # self.response.out.write('Result: ' + repr(result))
    self.response.write(hash)


class GetMeanSquaredError(MlWebserviceBase):
  """Gets the Mean Squared Error of a trained model.
  """

  def post(self):
    model_id = self.request.get('id')
    result = service.trainedmodels().get(project=constants.PROJECT_ID,
                                         id=model_id).execute()
    self.response.headers['Content-Type'] = constants.OUTPUT_CONTENT_TYPE_TEXT
    self.response.out.write(result['modelInfo']['meanSquaredError'])


class GetPredictions(MlWebserviceBase):
  """Gets predictions for a matrix of features using a given a trained model.
  """

  def post(self):
    model_id = self.request.get('id')
    data = self.request.get('data').encode(constants.UTF8_ENCODING_LABEL)
    logging.info('Features received for evaluation: %s', data)
    stream_from_str = StringIO.StringIO(data)
    reader = csv.reader(stream_from_str, delimiter=',')
    results = []
    for row in reader:
      values = []
      for column in row:
        values.append(column)
      payload = {'input': {'csvInstance': values}}
      result = service.trainedmodels().predict(project=constants.PROJECT_ID,
                                               id=model_id,
                                               body=payload).execute()
      results.append(float(result['outputValue']))
    # Write to the browser.
    self.response.headers['Content-Type'] = constants.OUTPUT_CONTENT_TYPE_TEXT
    self.response.out.write(json.dumps(results))


app = webapp2.WSGIApplication(
    [
        ('/train', TrainHandler),
        ('/msq_error', GetMeanSquaredError),
        ('/predict_multi', GetPredictions)
    ],
    debug=True)
