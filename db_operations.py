import mysql.connector as conn
from utils import *
from configparser import ConfigParser


FEATURE_SET_CODES = [216, 108, 38, 4610, 13824, 27648]

config = ConfigParser()
config.read_file(open('config.ini'))
data_config = config['data']

DESCRIPTORS_PATH = data_config['descriptors_path']


class DBOps:
    CMD_FIND_FEATURE_SET = '''
    SELECT descriptor_code, file_name 
    FROM descriptors
    WHERE costumeFK = %(costume_id)s 
        AND color_model = %(color_model)s 
        AND similarity_metric = %(similarity_metric)s
        AND model_name = %(model_name)s;'''

    CMD_FIND_THRESHOLDS = '''
    SELECT descriptor_code, threshold 
    FROM descriptors
    WHERE costumeFK = %(costume_id)s 
        AND color_model = %(color_model)s 
        AND similarity_metric = %(similarity_metric)s
        AND model_name = %(model_name)s
        AND threshold IS NOT NULL;'''

    CMD_SUPERTYPE_LOOKUP = '''
    SELECT costumeID, costume_parentFK 
    FROM costumes;'''

    CMD_LIST_FEATURE_SETS = '''
    SELECT DISTINCT costumeFK 
    FROM descriptors
    WHERE color_model = %(color_model)s 
        AND similarity_metric = %(similarity_metric)s
        AND model_name = %(model_name)s;'''

    CMD_LIST_SAMPLE_GROUPS = '''
    SELECT u.costumeFK, p.feature_code, COUNT(p.file_name), GROUP_CONCAT(p.file_name) 
    FROM patches AS p
    JOIN uploads AS u ON u.uploadID=p.uploadFK
    WHERE p.is_usable=1
    GROUP BY p.feature_code, u.costumeFK;'''

    CMD_COSTUME_INFO = '''
    SELECT t.name_d, t.description_d, r.name_d, c.name_d 
    FROM costumes AS t
    JOIN regions AS r ON r.regionID=t.regionFK
    JOIN cantons AS c ON c.cantonID=r.cantonFK
    WHERE t.costumeID=%(costume_id)s;'''

    def __init__(self, host, db_name, username, password):
        self.db = conn.connect(host=host, database=db_name, user=username, passwd=password)

    def class_lookup_dict(self):
        cursor = self.db.cursor()
        cursor.execute(DBOps.CMD_SUPERTYPE_LOOKUP)
        # map to superclass if one exists, otherwise map to same class
        return {c[0]: c[1] if c[1] else c[0] for c in cursor.fetchall()}

    def costume_info(self, costume_id):
        cursor = self.db.cursor()
        cursor.execute(DBOps.CMD_COSTUME_INFO, {'costume_id': costume_id})
        return cursor.fetchone()

    def sample_groups(self):
        cursor = self.db.cursor()
        cursor.execute(DBOps.CMD_LIST_SAMPLE_GROUPS)
        # set of samples
        data = {}  # cls -> {feature_code -> [cutout_path]}, {feature_code -> [cutout_path]}, ...
        for costume_id, feature_code, num_of_files, file_names in cursor.fetchall():
            if costume_id not in data:
                data[costume_id] = {}
            if num_of_files > 1:
                data[costume_id][feature_code] = file_names.split(',')
            elif num_of_files == 1:
                data[costume_id][feature_code] = [file_names]
        return data

    def load_thresholds(self, costume_id, color_model, similarity_metric, model_name='BODY25'):
        cursor = self.db.cursor()
        cursor.execute(DBOps.CMD_FIND_THRESHOLDS, {
            'costume_id': costume_id, 'color_model': color_model,
            'similarity_metric': similarity_metric, 'model_name': model_name
        })
        return [th for fc, th in cursor.fetchall()]

    def load_feature_set(self, costume_id, color_model, similarity_metric, model_name='BODY25',
                         descriptors_path=DESCRIPTORS_PATH):
        cursor = self.db.cursor()
        cursor.execute(DBOps.CMD_FIND_FEATURE_SET, {
            'costume_id': costume_id, 'color_model': color_model, 
            'similarity_metric': similarity_metric, 'model_name': model_name
            })
        data = {fc: file_name for fc, file_name in cursor.fetchall()}
        return [np.load('%s/%s.npy' % (descriptors_path, data[fc]))
                if fc in data else None for fc in FEATURE_SET_CODES]

    def feature_sets_available(self, color_model, similarity_metric, model_name='BODY25'):
        cursor = self.db.cursor()
        cursor.execute(DBOps.CMD_LIST_FEATURE_SETS, {
            'color_model': color_model,
            'similarity_metric': similarity_metric,
            'model_name': model_name})
        return [c[0] for c in cursor.fetchall()]
