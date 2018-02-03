import os.path
import pickle
import numpy as np

# Simple class to manage creating a dataset from a comma delimited file
class CSVLoader(object):

    def __init__(self, csvfile):

        self.savename = csvfile.split(".")[0] + ".data"
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

        self.data = np.array(rows)


    # There needs to be a method to correctly extract rows from the raw file lines.
    # This can return a list or a numpy array
    def extract(self, row): raise NotImplementedError


# CSV loader specific to the HMDA dataset
class HMDALoader(CSVLoader):

    # Constructor takes the base csvfile as well as a list of fields you want to look at.
    def __init__(self, csvfile, feature_fields, labelfield):

        if labelfield in feature_fields:
            raise AttributeError("Predicted label shouldn't be in your feature fields")

        # Build a logical index of all the headers that were chosen for easy retrieval of correct fields later
        with open(csvfile, 'r') as readfile:
            self.header = readfile.readline()
            self.headerindex = np.zeros([1, len(self.header)])
            try:
                for field in feature_fields:
                    self.headerindex[feature_fields.index(field)] = 1
            except ValueError:
                raise ValueError("Make sure your field strings are correct.")

        super(HMDALoader, self).__init__(csvfile)


    # Implementing an extract method to get all the fields I want.
    def extract(self, row):
        return self.transform(np.array(row.split(","))[self.headerindex])

    # Applying feature transformations
    def transform(self, data_array): raise NotImplementedError


# The specific loader I'm using for this problem.
# Here is where I bound my features and transform them into vectors appropriate for my modeling approach.
class WayneLoanApprovalLoader(HMDALoader):

    def __init__(self, csvfile):

        feature_fields = ["tract_to_msamd_income",
                          "rate_spread",
                          "population",
                          "minority_population",
                          "number_of_owner_occupied_units",
                          "number_of_1_to_4_family_units",
                          "loan_amount_000s",
                          "hud_median_family_income",
                          "applicant_income_000s",
                          "state_name",
                          "state_abbr",
                          "sequence_number",
                          "respondent_id",
                          "purchaser_type_name",
                          "property_type_name",
                          "preapproval_name",
                          "owner_occupancy_name",
                          "msamd_name",
                          "loan_type_name",
                          "loan_purpose_name",
                          "lien_status_name",
                          "hoepa_status_name",
                          "edit_status_name",
                          "denial_reason_name_3",
                          "denial_reason_name_2",
                          "denial_reason_name_1",
                          "county_name",
                          "co_applicant_sex_name",
                          "co_applicant_race_name_5",
                          "co_applicant_race_name_4",
                          "co_applicant_race_name_3",
                          "co_applicant_race_name_2",
                          "co_applicant_race_name_1",
                          "co_applicant_ethnicity_name",
                          "census_tract_number",
                          "as_of_year",
                          "application_date_indicator",
                          "applicant_sex_name",
                          "applicant_race_name_5",
                          "applicant_race_name_4",
                          "applicant_race_name_3",
                          "applicant_race_name_2",
                          "applicant_race_name_1",
                          "applicant_ethnicity_name",
                          "agency_name",
                          "agency_abbr"
                          ]

        labelfield =  "action_taken_name"

        super(WayneLoanApprovalLoader, self).__init__(csvfile, feature_fields, labelfield)

    # For my very specific use case, the field specific transforms I want to apply are in here.
    def transform(self, data_array):


