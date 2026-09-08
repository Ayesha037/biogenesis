from biogenesis.retrieval.pubmed_client import PubMedClient

SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal><Title>Journal of Testing</Title></Journal>
        <ArticleTitle>Effects of Drug A on Condition B</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Condition B is common.</AbstractText>
          <AbstractText Label="RESULTS">Drug A reduced symptoms significantly.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
        </AuthorList>
        <ArticleIdList>
          <ArticleId IdType="doi">10.1000/example.doi</ArticleId>
        </ArticleIdList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Drug A</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="pubmed"><Year>2022</Year></PubMedPubDate>
      </History>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_efetch_xml_extracts_expected_fields():
    papers = PubMedClient._parse_efetch_xml(SAMPLE_XML)
    assert len(papers) == 1
    p = papers[0]
    assert p.pmid == "12345678"
    assert p.title == "Effects of Drug A on Condition B"
    assert "Drug A reduced symptoms" in p.abstract
    assert p.authors == ["Jane Smith"]
    assert p.doi == "10.1000/example.doi"
    assert "Drug A" in p.mesh_terms
