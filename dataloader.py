import os.path
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale

# Simple class to manage creating a dataset from a comma delimited file
class FileLoader(object):

    def __init__(self, savename, csvfile, delimiter):

        self.savename = savename + ".data"
        self.delimiter = delimiter
        self.data = None

        # If a saved data file already exists just load that instead of processing ingestion again.
        if self.check_exist(self.savename):
            with open(self.savename, 'rb') as handle:
                return pickle.load(handle)
        else:
            self.data = self.ingest(csvfile)
            with open(self.savename, 'wb') as handle:
                pickle.dump(self, handle)


    # Check if saved data array already exists.
    def check_exist(self, savename):
        os.path.isfile(savename)

    def ingest(self, csvfile):

        rows = []

        with open(csvfile, 'r') as readfile:

            header = readfile.readline() # Skipping first header line

            for line in readfile:
                rows.append(self.extract(line))

        return self.trawl(np.array(rows))


    # Data wide transformations, like scaling or standardization
    def trawl(self, data): raise NotImplementedError

    # There needs to be a method to correctly extract rows from the raw file lines.
    # This can return a list or a numpy array
    def extract(self, row): raise NotImplementedError


# File loader specific to the HMDA dataset
class HMDALoader(FileLoader):

    # Constructor takes the base csvfile as well as a list of fields you want to look at.
    def __init__(self, savename, csvfile, delimiter, feature_fields, labelfield):

        self.delimiter = delimiter

        if labelfield in feature_fields:
            raise AttributeError("Predicted label shouldn't be in your feature fields")

        self.features = feature_fields
        self.label = labelfield
        # Build a logical index of all the headers that were chosen for easy retrieval of correct fields later
        with open(csvfile, 'r') as readfile:
            self.header = readfile.readline().strip().split(self.delimiter)
            self.headerindex = np.zeros([len(self.header)]).astype('bool')
            self.labelindex = np.zeros([len(self.header)]).astype('bool')
            try:
                for field in feature_fields:
                    self.headerindex[self.header.index(field)] = 1

                self.labelindex[self.header.index(labelfield)] = 1

            except ValueError:
                raise ValueError("Make sure your field strings are correct.")

        super(HMDALoader, self).__init__(savename, csvfile, delimiter)






# The specific loader I'm using for this problem.
# Here is where I bound my features and transform them into vectors appropriate for my modeling approach.
# I'm also adding a date index to let me explore some hypotheses about time...
class WayneLoanApprovalLoader(HMDALoader):

    def __init__(self, savename, csvfile, feature_fields_map = None):

        delimiter = "\t" # Assuming I'm using a .tsv file

        # Defining a map that creates binary labels from 'action_taken'
        self.label_collapse_map = {"Application approved but not accepted" : 1,
                                   "Application denied by financial institution" : 0,
                                   "Loan originated" : 1,
                                   "Preapproval request denied by financial institution" : 0
                                   }

        # holder state for items that are useful for feature importance tasks
        self.features_to_vector_idx = {}
        self.categoricals = {}
        self.vector_headers = None

        # Important val for indexing based on time.
        self.date_column = []

        # typing map from fields to types I want to transform to
        if feature_fields_map is None:
            self.feature_fields_map = {"tract_to_msamd_income" : 'indicator',
                              "rate_spread" : 'indicator',
                              "population" : 'indicator',
                              "minority_population" : 'indicator',
                              "number_of_owner_occupied_units" : 'indicator',
                              "number_of_1_to_4_family_units" : 'indicator',
                              "loan_amount_000s" : 'indicator',
                              "hud_median_family_income" : 'indicator',
                              "applicant_income_000s" : 'indicator',
                              "property_type_name" : 'categorical',
                              "preapproval_name" : 'categorical',
                              "owner_occupancy_name" : 'categorical',
                              "loan_type_name" : 'categorical',
                              "lien_status_name" : 'categorical',
                              "hoepa_status_name" : 'categorical',
                              "co_applicant_sex_name" : 'categorical',
                              "co_applicant_race_name_1" : 'categorical',
                              "co_applicant_ethnicity_name" : 'categorical',
                              "applicant_sex_name" : 'categorical',
                              "applicant_race_name_1" : 'categorical',
                              "applicant_ethnicity_name" : 'categorical',
                              "agency_name" : 'categorical'}
        else:
            self.feature_fields_map = feature_fields_map

        labelfield = "action_taken_name"

        super(WayneLoanApprovalLoader, self).__init__(savename, csvfile, delimiter, list(self.feature_fields_map.keys()), labelfield)

    # For my very specific use case, the field specific transforms I want to apply are in here.
    def trawl(self, data):

        def getColumnsNum(chunks):

            if chunks is None:
                return 0
            else:
                return chunks.shape[1] - 1


        chunks = None # I'll be making submatrices here that I'll stitch together at the very end.

        # Python doesn't have type matching as good as scala's... so here I am matching on a manually built map.
        for field in self.features:

            # If the field is a float field, then standardize the whole row.
            if self.feature_fields_map[field] == 'float64':
                column = [x if len(x) > 0 else 0 for x in data[:, self.features.index(field)] ] # Cleaning out empty vals
                chunk = scale(np.expand_dims(column, 1).astype('float64'))

                self.features_to_vector_idx[field] = getColumnsNum(chunks)  # To let us know where in the vec a feature is

                if chunks is None:
                    chunks = chunk
                else:
                    chunks = np.concatenate((chunks, chunk), 1)



            # If the field is categorical, then replace the column with onehot encodings.
            elif self.feature_fields_map[field] == 'categorical':
                lb = LabelBinarizer()
                lb.fit(data[:, self.features.index(field)])
                lb.classes_ = sorted(lb.classes_)
                chunk = lb.transform(data[:, self.features.index(field)])

                self.features_to_vector_idx[field] = getColumnsNum(chunks) # To let us know where in the vec a feature is
                self.categoricals[field] = lb.classes_ # To let us know what categories were encoded

                if chunks is None:
                    chunks = chunk
                else:
                    chunks = np.concatenate((chunks, chunk), 1)

            elif self.feature_fields_map[field] == 'indicator':

                double_column = [[x, 0] if len(x) > 0 else [0, 1] for x in data[:, self.features.index(field)]]
                double_column = np.array(double_column)
                indicators = double_column[:, 1]
                vals = scale(double_column[:, 0])

                if chunks is None:
                    chunks = np.concatenate((np.expand_dims(vals, 1), np.expand_dims(indicators, 1)), 1)
                else:
                    chunks = np.concatenate((chunks, np.expand_dims(vals, 1), np.expand_dims(indicators,1)), 1)

        # Save the labelfield for last.
        labels = np.expand_dims(data[:, -1], 1)
        chunks = np.concatenate((chunks, labels), 1)

        # Putting together the full list of the new features we just transformed
        # These can later be aligned with weight vectors from models and whatnot
        # which is generally useful for model explainability
        headers = []
        for field in self.features:
            if self.feature_fields_map[field] == 'float64' :
                headers.append(field)
            elif self.feature_fields_map[field] == 'categorical':
                categories = list(self.categoricals[field])
                if len(categories) > 2:
                    headers += [field + ": " + category for category in list(self.categoricals[field])]
                elif len(categories) == 2:
                    headers += [categories[0] + "/" + categories[1]]
                else:
                    headers += [categories[0]]
            elif self.feature_fields_map[field] == 'indicator':
                headers.append(field)
                headers.append(field + "_indicator")

        self.vector_headers = headers

        return chunks.astype('float64')

    # Implementing an extract method to get all the fields I want.
    def extract(self, textrow):

        row = np.array(textrow.split(self.delimiter)[:-1])
        features = row[self.headerindex] # [np.array of mixed type]
        labels = [self.label_collapse_map[row[self.labelindex][0]]]  # [int]

        # Compiling my data column row by row
        self.date_column.append(row[self.header.index('as_of_year')])

        # Returning features with a label value at the very end.
        return np.concatenate((features, labels))  # np.array row of mixed type

    # This method lets you quickly index by date to pull out chunks of data.
    def get_dates(self, datelist):

        index = np.zeros([self.data.shape[0]]).astype('bool')

        for date in datelist:

            index += np.array(self.date_column).astype('int') == int(date)

        return self.data[index, :]


