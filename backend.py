from flask import Flask, request, render_template
from configparser import ConfigParser
from db_operations import DBOps
from uuid import uuid4
from classifier_worker import classify


app = Flask(__name__)
app.logger.setLevel('DEBUG')
config = ConfigParser()
config.read_file(open('config.ini'))

db_config = config['database']
host_name = db_config.get('host')
db_name = db_config.get('name')
username = db_config.get('username')
password = db_config.get('password')

classifier_config = config['classifier']
similarity_model = classifier_config.get('similarity_metric')

data_config = config['data']
uploads_path = data_config.get('uploads_path')

db_ops = DBOps(host_name, db_name, username, password)

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

class_lookup = db_ops.class_lookup_dict()
class_names = {c: db_ops.costume_info(c)[0] for c in class_lookup.keys()}

class_lookup[-1] = -1  # rejection class, insert only after names for existing classes are filled
class_names[-1] = 'Unknown costume'


@app.route('/')
def home():
    return render_template('index.html')


def extension(filename):
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else None


@app.route('/predict', methods=['POST'])
def predict():
    app.logger.debug('new request')

    if 'file' not in request.files:
        app.logger.error('no file part')
        return 'Not a file'
    file = request.files['file']
    if file.filename == '':
        return 'File not valid'

    ext = extension(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        return 'File type not allowed'

    filename = uploads_path + str(uuid4())
    file.save('%s.%s' % (filename, ext))

    app.logger.debug('%s: image saved' % filename)

    promise = classify.apply_async((filename, ext))
    app.logger.debug('task sent')
    scores = promise.get()
    scores = {int(c): s for c, s, in scores.items()}  # Celery seems to stringify dict keys, convert back to int

    app.logger.debug(scores)

    supertype_scores = reduce_to_supertype_and_sort(scores)
    named_supertype_scores = [(i, class_names[i], s) for i, s in supertype_scores]
    
    app.logger.debug('%s: results ready')

    return render_template('results.html', scores=named_supertype_scores[:10])


def reduce_to_supertype_and_sort(class_scores):
    sorted_class_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse='mse' in similarity_model)
    # recurring supertypes will be overwritten with better scores then sort by best score
    reduced = {class_lookup[cl] if cl in class_lookup else cl: sc for cl, sc in sorted_class_scores}
    return sorted(reduced.items(), key=lambda x: x[1], reverse='mse' not in similarity_model)


@app.route('/costumes/<costume_id>', methods=['GET'])
def costume_info(costume_id):
    if costume_id == '-1':
        return render_template('costume.html', name='Unknown costume', region='--', canton='--',
                               description="We couldn't recognize the costume in the photo.")

    costume_name, costume_description, region_name, canton_name = db_ops.costume_info(costume_id)
    return render_template('costume.html', name=costume_name, description=costume_description, 
                           region=region_name, canton=canton_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
