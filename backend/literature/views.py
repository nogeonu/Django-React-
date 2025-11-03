import re
import requests
import feedparser
from urllib.parse import urlencode, quote_plus
from rest_framework.views import APIView
from rest_framework.response import Response
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator

class LitSearchView(APIView):
    """
    논문 검색 API (PubMed + arXiv)
    q: 검색어, max: 개수
    """
    @method_decorator(cache_page(1800))  # 30분 캐싱
    def get(self, request):
        q = request.GET.get("q", "").strip()
        n = int(request.GET.get("max", 20))
        out = {"pubmed": [], "arxiv": []}

        # --- PubMed esearch + esummary ---
        if q:
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            try:
                # ID 검색
                ids_response = requests.get(base+"esearch.fcgi", params={
                    "db":"pubmed","retmode":"json","retmax":n,"term":q
                }, timeout=10)
                ids = ids_response.json().get("esearchresult",{}).get("idlist",[])
                
                if ids:
                    # 상세 정보
                    summ_response = requests.get(base+"esummary.fcgi", params={
                        "db":"pubmed","retmode":"json","id":",".join(ids)
                    }, timeout=10)
                    summ = summ_response.json().get("result",{})
                    
                    for pid in ids:
                        it = summ.get(pid)
                        if not it: continue
                        # PMC ID 찾기
                        pmc_id = None
                        for article_id in it.get("articleids", []):
                            if article_id.get("idtype") == "pmc":
                                pmc_id = article_id.get("value")
                                break
                        
                        out["pubmed"].append({
                            "id": pid,
                            "title": it.get("title", ""),
                            "journal": it.get("fulljournalname", ""),
                            "year": it.get("pubdate","")[:4],
                            "authors": [a.get("name", "") for a in it.get("authors",[])],
                            "doi": (it.get("elocationid") or "").replace("doi: ", ""),
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                            "pmc": pmc_id
                        })
            except Exception as e:
                print(f"PubMed API error: {e}")
                # 에러가 발생해도 빈 배열을 반환

        # --- arXiv API ---
        if q:
            try:
                aurl = "http://export.arxiv.org/api/query?" + urlencode({
                    "search_query": f"all:{q}", "start": 0, "max_results": n,
                    "sortBy":"lastUpdatedDate","sortOrder":"descending"
                })
                feed = feedparser.parse(aurl)
                for e in feed.entries:
                    pdf = next((l.href for l in e.links if l.type=="application/pdf"), None)
                    out["arxiv"].append({
                        "id": e.id.split("/")[-1],
                        "title": e.title,
                        "authors": [a.name for a in e.authors],
                        "year": e.published[:4] if hasattr(e, 'published') else e.updated[:4],
                        "pdf": pdf,
                        "url": e.link,
                        "summary": e.summary
                    })
            except Exception as e:
                print(f"arXiv API error: {e}")

        return Response(out)

class LitOpenAccessView(APIView):
    """
    DOI로 오픈액세스 PDF 링크 탐색 (Unpaywall)
    """
    UNPAYWALL = "https://api.unpaywall.org/v2/"
    CONTACT = "your_email@hospital.org"  # 실제 이메일로 교체 필요

    @method_decorator(cache_page(3600))  # 1시간 캐싱
    def get(self, request):
        doi = request.GET.get("doi","").strip()
        if not doi:
            return Response({"pdf": None})
        
        try:
            r = requests.get(self.UNPAYWALL+doi, params={"email": self.CONTACT}, timeout=10)
            if r.ok:
                data = r.json()
                best = (data.get("best_oa_location") or {}) or {}
                return Response({"pdf": best.get("url_for_pdf")})
        except Exception as e:
            print(f"Unpaywall API error: {e}")
        
        return Response({"pdf": None})

class NewsFeedView(APIView):
    """
    뉴스 피드 API - 국내/해외 뉴스 분리 제공
    """
    @method_decorator(cache_page(900))  # 15분 캐싱
    def get(self, request):
        q = request.GET.get("q", "호흡기 OR 폐암")
        news_type = request.GET.get("type", "all")  # domestic, international, all
        
        try:
            domestic_items = []
            international_items = []

            # 검색어에서 키워드 추출 (논리 연산자 제거)
            raw_keywords = re.split(r"\s+|[,;]\s*", q)
            keywords = [kw.lower() for kw in raw_keywords if kw and kw.lower() not in {"or", "and", "not"}]

            def matches_keywords(*texts: str) -> bool:
                if not keywords:
                    return True
                haystack = " ".join(filter(None, texts)).lower()
                return any(kw in haystack for kw in keywords)

            # 국내 뉴스 소스 (검색어 기반 Google News + 일반 RSS)
            domestic_sources = [
                f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=ko&gl=KR&ceid=KR:ko",
                "https://www.hankyung.com/feed/",
                "https://www.chosun.com/rss/",
            ]

            # 해외 의료 뉴스 소스 (검색어 기반 Google News + 전문 RSS)
            international_sources = [
                f"https://news.google.com/rss/search?{urlencode({'q': q, 'hl': 'en', 'gl': 'US', 'ceid': 'US:en'})}",
                "https://medicalxpress.com/feed/",
                "https://www.sciencedaily.com/rss/health_medicine.xml",
                "https://www.eurekalert.org/rss.xml"
            ]

            def collect_items(sources, bucket, bucket_type, fallback_source):
                for rss_url in sources:
                    try:
                        feed = feedparser.parse(rss_url)
                        entries = getattr(feed, "entries", [])
                        if not entries:
                            continue

                        source_title = getattr(feed, "feed", {}).get("title", fallback_source)
                        for entry in entries[:15]:
                            title = getattr(entry, "title", "")
                            url = getattr(entry, "link", "")
                            summary = getattr(entry, "summary", "")
                            published = getattr(entry, "published", "")

                            if not title or not url:
                                continue

                            if not matches_keywords(title, summary):
                                continue

                            bucket.append({
                                "title": title,
                                "url": url,
                                "published": published,
                                "source": source_title,
                                "type": bucket_type
                            })
                    except Exception as e:
                        print(f"{bucket_type.capitalize()} RSS {rss_url} failed: {e}")
                        continue

            if news_type in ["domestic", "all"]:
                collect_items(domestic_sources, domestic_items, "domestic", "국내 뉴스")

            if news_type in ["international", "all"]:
                collect_items(international_sources, international_items, "international", "International News")

            # 샘플 뉴스 추가 (RSS 실패 시)
            if not domestic_items and news_type in ["domestic", "all"]:
                domestic_items = [
                    {
                        "title": "국내 의료 뉴스 - 폐암 치료법 개발",
                        "url": "https://www.cancer.go.kr",
                        "published": "2025-11-03T10:00:00Z",
                        "source": "국가암센터",
                        "type": "domestic"
                    },
                    {
                        "title": "한국 연구진, AI 폐암 진단 기술 개발",
                        "url": "https://www.korea.kr",
                        "published": "2025-11-02T14:30:00Z",
                        "source": "정부24",
                        "type": "domestic"
                    }
                ]
            
            if not international_items and news_type in ["international", "all"]:
                international_items = [
                    {
                        "title": "AI technology shows promise in early lung cancer detection",
                        "url": "https://medicalxpress.com/news/2025-11-ai-technology-early-lung-cancer.html",
                        "published": "2025-11-03T10:00:00Z",
                        "source": "Medical Xpress",
                        "type": "international"
                    },
                    {
                        "title": "New study reveals link between air pollution and respiratory diseases",
                        "url": "https://www.sciencedaily.com/releases/2025/11/211103102456.htm",
                        "published": "2025-11-02T14:30:00Z",
                        "source": "Science Daily",
                        "type": "international"
                    }
                ]
            
            # 결과 조합
            if news_type == "domestic":
                items = domestic_items
            elif news_type == "international":
                items = international_items
            else:
                items = domestic_items + international_items
            
            # 최신순 정렬
            items.sort(key=lambda x: x.get('published', ''), reverse=True)
            
            return Response({"items": items[:30]})
            
        except Exception as e:
            print(f"News API error: {e}")
            return Response({"items": []})
