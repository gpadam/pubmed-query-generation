
#Function to test queries in PubMed and calculate statistics
#Note: you must provide your information for PubMed's API to run

#requred packages
!pip3 install biopython
from Bio import Entrez
!pip install datasets evaluate transformers[sentencepiece]
!pip install accelerate
#!git config --global user.email "USER EMAIL"
#!git config --global user.name "USER NAME"

def query_evaluation(query, altquery, GS_query, included_studies):
    #Run title only query
    #Entrez.email = "USER EMAIL HERE"
    handle = Entrez.esearch(db="pubmed", retmax=0, term=query)
    record = Entrez.read(handle)
    handle.close()
    mcount_query = float(record["Count"])

    #run title and KQ query
    handle = Entrez.esearch(db="pubmed", retmax=0, term=altquery)
    record = Entrez.read(handle)
    handle.close()
    mcount_altquery = float(record["Count"])

    #run original query
    handle = Entrez.esearch(db="pubmed", retmax=0, term=GS_query)
    record = Entrez.read(handle)
    handle.close()
    gsmcount = float(record["Count"])

    #run final included PMIDs
    handle = Entrez.esearch(db="pubmed", retmax=0, term=included_studies)
    record = Entrez.read(handle)
    handle.close()
    pmidsmcount = float(record["Count"])

    #Run title query AND original query
    combquery_title = " AND ".join([query,GS_query])
    handle = Entrez.esearch(db="pubmed", retmax=0, term=combquery_title)
    record = Entrez.read(handle)
    handle.close()
    combcount_title = float(record["Count"])

    #Run title and KQ query AND original query
    combquery_tikq = " AND ".join([altquery,GS_query])
    handle = Entrez.esearch(db="pubmed", retmax=0, term=combquery_tikq)
    record = Entrez.read(handle)
    handle.close()
    combcount_tikq = float(record["Count"])

    #Run title query AND included PMIDs
    missed_query_title = " AND ".join([included_studies,query])
    handle = Entrez.esearch(db="pubmed", retmax=0, term=missed_query_title)
    record = Entrez.read(handle)
    handle.close()
    true_positive_title = float(record["Count"])

    #Run title and KQ query AND included PMIDs
    missed_query_tikq = " AND ".join([included_studies,altquery])
    handle = Entrez.esearch(db="pubmed", retmax=0, term=missed_query_tikq)
    record = Entrez.read(handle)
    handle.close()
    true_positive_tikq = float(record["Count"])

    #Calculate title only statistics
    pmids_count = float(len(included_studies.split()))
    if mcount_query >0 and true_positive_title >0:
      sensitivity_title = round(((true_positive_title)/(pmids_count)*100),2)
      precision_title = ((true_positive_title)/(mcount_query))
      NNR_title = round((1/precision_title),0)
    else:
      sensitivity_title = 0
      precision_title = 0
      NNR_title = "NA"

    #Calculate title and KQ statistics
    if mcount_altquery >0 and true_positive_tikq >0:
      sensitivity_tikq = round(((true_positive_tikq)/(pmids_count)*100),2)
      precision_tikq = ((true_positive_tikq)/(mcount_altquery))
      NNR_tikq = round((1/precision_tikq),0)
    else:
      sensitivity_tikq = 0
      precision_tikq = 0
      NNR_tikq = "NA"

    #return values
    return(mcount_query, mcount_altquery, gsmcount, pmidsmcount, combcount_title, combcount_tikq, true_positive_title, true_positive_tikq, true_positive_title, sensitivity_title, precision_title, NNR_title, true_positive_tikq, sensitivity_tikq, precision_tikq, NNR_tikq)


#Function for missed studies analysis

def missed(altquery, included_studies):
    #Entrez.email = "USER EMAIL"
    combquery_AND = " AND ".join([included_studies, altquery])
    handle = Entrez.esearch(db="pubmed", retmax=4000, term=combquery_AND)
    record = Entrez.read(handle)
    handle.close()
    found = record["IdList"]

    #Entrez.email = "USER EMAIL"
    combquery_NOT = " NOT ".join([included_studies, altquery])
    handle = Entrez.esearch(db="pubmed", retmax=4000, term=combquery_NOT)
    record = Entrez.read(handle)
    handle.close()
    missed = record["IdList"]

    #retrieve the results
    if len(found) >0:
      #Entrez.email = "USER EMAIL"
      handle = Entrez.efetch(db="pubmed", id=found, rettype="xml")
      found_xml = Entrez.read(handle)
      handle.close()
    else:
      found_xml = "NA"

    #Entrez.email = "USER EMAIL"
    if len(missed) >0:
     handle = Entrez.efetch(db="pubmed", id=missed, rettype="xml")
     missed_xml = Entrez.read(handle)
     handle.close()
    else:
      missed_xml = "NA"

    return(found_xml,missed_xml)
