# run: python api.py

from crypt import methods
from flask import Flask, request, make_response, send_file
from flasgger import Swagger
from stemming.porter2 import stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from io import BytesIO
import time
import zipfile
import pickle
import pandas as pd
import numpy as np

with open("./pickles/forest.pkl", "rb") as model_file: # model to predict from
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app) # http://127.0.0.1:5050/apidocs/

# IRIS API

@app.route("/iris", methods=["GET"]) # http://127.0.0.1:5050/iris?s_length=5.7&s_width=5.6&p_length=4.3&p_width=7.8
def iris():
    """
    Returns a prediction of iris. 
    ---
    parameters:
        - name: s_length
          in: query
          type: number
          required: true
        - name: s_width
          in: query
          type: number
          required: true
        - name: p_length
          in: query
          type: number
          required: true
        - name: p_width
          in: query
          type: number
          required: true
    responses:
        200:
            description: OK
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction)

@app.route("/iris_file", methods=["POST"]) # http://127.0.0.1:5050/iris_file
def iris_input():
    """
    Returns a prediction of iris given a csv file. 
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: OK
    """
    input_file = pd.read_csv(request.files.get("input_file"), header=None) # form-data
    prediction = model.predict(input_file)
    return str(prediction)

# TEXT API

def clean_text(txt):
  if txt:
    cleaned = " ".join(txt.split()) # remove whitespaces
    reduced_text = [stem(word) for word in cleaned.split()] # stemming
    return " ".join(reduced_text)
  else:
    return txt

@app.route("/cluster", methods=["POST"])
def cluster():
  """
  Returns clusters given a csv file. Can pass the name of the text column as query param.
  ---
  parameters:
      - name: dataset
        in: formData
        type: file
        required: true
      - name: col
        in: query
        type: string
        required: false
  responses:
      200:
          description: OK
  """
  data = pd.read_csv(request.files["dataset"])
  unstructure = "text"
  if "col" in request.args:
    unstructure = request.args.get("col")
  no_of_clusters = 2
  if "no_of_clusters" in request.args:
    no_of_clusters = int(request.args.get("no_of_clusters"))
  data = data.fillna("NULL")
  data["clean_sum"] = data[unstructure].apply(clean_text)
  vectorizer = CountVectorizer(analyzer="word", stop_words="english") # matrix with words frequency
  counts = vectorizer.fit_transform(data["clean_sum"]) # get only the clean column
  kmeans = KMeans(n_clusters=no_of_clusters)
  data["cluster_num"] = kmeans.fit_predict(counts)
  data = data.drop(["clean_sum"], axis=1)
  output = BytesIO()
  writer = pd.ExcelWriter(output, engine="xlsxwriter")
  data.to_excel(writer, sheet_name="clusters", encoding="utf-8", index=False)
  clusters = []
  for i in range(np.shape(kmeans.cluster_centers_)[0]): # new sheet to display top 10 keywords for each text
    data_cluster = pd.concat([pd.Series(vectorizer.get_feature_names()), pd.DataFrame(kmeans.cluster_centers_[i])], axis=1)
    data_cluster.columns = ['keywords', 'weights']
    data_cluster = data_cluster.sort_values(by=['weights'], ascending=False)
    data_clust = data_cluster.head(n=10)['keywords'].tolist()
    clusters.append(data_clust)
  pd.DataFrame(clusters).to_excel(writer, sheet_name='top_keywords', encoding='utf-8')
  # pivot
  data_pivot = data.groupby(['cluster_num'], as_index=False).size()
  data_pivot.name = 'size'
  data_pivot = data_pivot.reset_index()
  data_pivot.to_excel(writer, sheet_name='cluster_report', encoding='utf-8', index=False)
  # insert chart
  workbook = writer.book
  worksheet = writer.sheets['cluster_report']
  chart = workbook.add_chart({'type': 'column'})
  chart.add_series({ 'values': '=cluster_report!$C$2:$C'+str(no_of_clusters+1) })
  worksheet.insert_chart('D2', chart)
    
  writer.save()
  memory_file = BytesIO()
  with zipfile.ZipFile(memory_file, "w") as zf:
    names = ["cluster_output.xlsx"]
    files = [output]
    for i in range(len(files)):
      data = zipfile.ZipInfo(names[i])
      data.date_time = time.localtime(time.time())
      data.compress_type = zipfile.ZIP_DEFLATED
      zf.writestr(data, files[i].getvalue())
  memory_file.seek(0)
  response = make_response(send_file(memory_file, as_attachment=True, attachment_filename="cluster_output.zip"))
  response.headers["Content-Disposition"] = "atachment;filename=cluster_output.zip"
  response.headers["Access-Control-Allow-Origin"] = "*"
  return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)

