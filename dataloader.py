import os.path
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale

# Simple class to manage creating a dataset from a comma delimited file
class FileLoader(object):

    def __init__(self, csvfile, delimiter):

        self.savename = csvfile.split(".")[0] + ".data"
        self.delimiter = delimiter
        self.data = None

        # If a saved data file already exists just load that instead of processing ingestion again.
        if self.check_exist(self.savename):
            with open(self.savename, 'rb') as handle:
                return pickle.load(handle)
        else:
            self.ingest(csvfile)



    # Check if saved data array already exists.
    def check_exist(self, savename):
        os.path.isfile(savename)

    def ingest(self, csvfile):

        rows = []

        with open(csvfile, 'r') as readfile:

            header = readfile.readline() # Skipping first header line

            for line in readfile:
                rows.append(self.extract(line))

        self.data = self.scale(np.array(rows))


    # Data wide transformations, like scaling or standardization
    def trawl(self, data): raise NotImplementedError

    # There needs to be a method to correctly extract rows from the raw file lines.
    # This can return a list or a numpy array
    def extract(self, row): raise NotImplementedError


# File loader specific to the HMDA dataset
class HMDALoader(FileLoader):

    # Constructor takes the base csvfile as well as a list of fields you want to look at.
    def __init__(self, csvfile, delimiter, feature_fields, labelfield):

        if labelfield in feature_fields:
            raise AttributeError("Predicted label shouldn't be in your feature fields")

        self.features = feature_fields
        self.label = labelfield
        # Build a logical index of all the headers that were chosen for easy retrieval of correct fields later
        with open(csvfile, 'r') as readfile:
            self.header = readfile.readline().split(self.delimiter)
            self.headerindex = np.zeros([1, len(self.header)])
            self.labelindex = np.zeros([1, len(self.header)])
            try:
                for field in feature_fields:
                    self.headerindex[self.header.index(field)] = 1

                self.labelindex[self.header.index(labelfield)] = 1

            except ValueError:
                raise ValueError("Make sure your field strings are correct.")

        super(HMDALoader, self).__init__(csvfile, delimiter)


    # Implementing an extract method to get all the fields I want.
    def extract(self, row):
        features = np.array(row.split(self.delimiter))[self.headerindex]
        labels = np.array(row.split(self.delimiter))[self.labelindex]

        #Returning features with a label value at the very end.
        return np.concatenate((features, labels))



# The specific loader I'm using for this problem.
# Here is where I bound my features and transform them into vectors appropriate for my modeling approach.
class WayneLoanApprovalLoader(HMDALoader):

    def __init__(self, csvfile):

        delimiter = "\t" # Assuming I'm using a .tsv file
        self.features_to_vector_idx = {}
        self.categoricals = {}
        self.feature_fields_map = {"tract_to_msamd_income" : 'float64',
                          "rate_spread" : 'float64',
                          "population" : 'float64',
                          "minority_population" : 'float64',
                          "number_of_owner_occupied_units" : 'float64',
                          "number_of_1_to_4_family_units" : 'float64',
                          "loan_amount_000s" : 'float64',
                          "hud_median_family_income" : 'float64',
                          "applicant_income_000s" : 'float64',
                          "property_type_name" : 'categorical',
                          "preapproval_name" : 'categorical',
                          "owner_occupancy_name" : 'categorical',
                          "loan_type_name" : 'categorical',
                          "lien_status_name" : 'categorical',
                          "hoepa_status_name" : 'categorical',
                          "co_applicant_sex_name" : 'categorical',
                          "co_applicant_race_name_1" : 'categorical',
                          "co_applicant_ethnicity_name" : 'categorical',
                          "as_of_year" : 'float64',
                          "applicant_sex_name" : 'categorical',
                          "applicant_race_name_1" : 'categorical',
                          "applicant_ethnicity_name" : 'categorical',
                          "agency_name" : 'categorical'}

        labelfield = "action_taken_name"

        super(WayneLoanApprovalLoader, self).__init__(csvfile, delimiter, list(self.feature_fields_map.keys()), labelfield)

    # For my very specific use case, the field specific transforms I want to apply are in here.
    def trawl(self, data):

        def getColumnsNum(chunks):

            if len(chunks) == 0:
                return 0
            else:
                sum = 0
                for chunk in chunks:
                    sum += chunk.shape[1]

                return sum


        chunks = [] # I'll be making submatrices here that I'll stitch together at the very end.

        # Python doesn't have type matching as good as scala's... so here I am matching on a manually built map.
        for field in self.features:

            # If the field is a float field, then standardize the whole row.
            if self.feature_fields_map[field] == 'float64':
                chunk = scale(np.expand_dims(self.data[:, self.features.index(field)], 1))

                self.features_to_vector_idx[field] = getColumnsNum(chunks)  # To let us know where in the vec a feature is
                chunks.append(chunk)



            # If the field is categorical, then replace the column with onehot encodings.
            elif self.feature_fields_map[field] == 'categorical':
                lb = LabelBinarizer()
                lb.fit(self.data[:, self.features.index(field)])
                chunk = lb.transform(self.data[:, self.features.index(field)])

                self.features_to_vector_idx[field] = getColumnsNum(chunks) # To let us know where in the vec a feature is
                self.categoricals[field] = lb.classes_ # To let us know what categories were encoded
                chunks.append(chunk)




        # Save the labelfield for last.
        








