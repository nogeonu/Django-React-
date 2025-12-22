import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ExternalLink, FileText, Search, Loader2, Calendar, User, Building, Globe } from 'lucide-react';
import { apiRequest } from '@/lib/api';
import { useAuth } from '@/context/AuthContext';

interface PubMedPaper {
  id: string;
  title: string;
  journal: string;
  year: string;
  authors: string[];
  doi: string;
  url: string;
  pmc?: string;
}

interface ArXivPaper {
  id: string;
  title: string;
  authors: string[];
  year: string;
  pdf?: string;
  url: string;
  summary: string;
}

interface NewsItem {
  title: string;
  url: string;
  published: string;
  source: string;
  type?: 'domestic' | 'international';
}

interface Guideline {
  title: string;
  description: string;
  url: string;
}

interface DeptContent {
  title: string;
  description: string;
  defaultSearch: string;
  defaultNews: string;
  guidelines: Guideline[];
}

const departmentConfig: Record<string, DeptContent> = {
  '호흡기내과': {
    title: '지식 허브',
    description: '최신 논문, 가이드라인, 뉴스를 통해 폐암 진단 및 치료 정보를 확인하세요.',
    defaultSearch: 'lung cancer OR 폐암',
    defaultNews: '호흡기 OR 폐암',
    guidelines: [
      { title: 'NCCN 가이드라인', description: '미국 국립 종양 네트워크 폐암 진료 가이드라인', url: 'https://www.nccn.org/guidelines/category_1' },
      { title: '대한암학회', description: '대한암학회 암 진료 권고안', url: 'https://www.cancer.go.kr' },
      { title: 'WHO 분류', description: 'WHO 종양 분류 기준', url: 'https://www.iarc.who.int' },
      { title: '미국흉부외과학회', description: '흉부외과 관련 자료 및 가이드라인', url: 'https://www.aats.org' },
      { title: 'ESMO 가이드라인', description: '유럽 종양 내과 학회 임상 가이드라인', url: 'https://www.esmo.org/guidelines' },
      { title: 'ASCO 가이드라인', description: '미국 임상 종양학회 진료 가이드라인', url: 'https://www.asco.org/guidelines' },
      { title: '대한결핵 및 호흡기학회', description: '호흡기 질환 진료 가이드라인', url: 'https://www.lungkorea.org' },
      { title: '미국흉부학회', description: 'ATS 호흡기 질환 가이드라인', url: 'https://www.thoracic.org' },
      { title: '국립암센터', description: '국가 암 관리 및 연구', url: 'https://www.ncc.re.kr' },
    ]
  },
  '외과': {
    title: '지식 허브 (유방암)',
    description: '최신 논문, 가이드라인, 뉴스를 통해 유방암 진단 및 치료 정보를 확인하세요.',
    defaultSearch: 'Breast Cancer OR 유방암',
    defaultNews: '유방암 OR Breast Cancer',
    guidelines: [
      { title: '한국유방암학회', description: '한국유방암학회 진료 가이드라인', url: 'https://www.kbcs.or.kr' },
      { title: 'NCCN Breast Cancer', description: '미국 국립 종양 네트워크 유방암 가이드라인', url: 'https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1419' },
      { title: 'ASCO Breast Cancer', description: '미국 임상 종양학회 유방암 가이드라인', url: 'https://www.asco.org/practice-patients/guidelines/breast-cancer' },
      { title: 'ESMO Breast Cancer', description: '유럽 종양 내과 학회 유방암 가이드라인', url: 'https://www.esmo.org/guidelines/guidelines-by-topic/breast-cancer' },
      { title: 'GBCC', description: '세계 유방암 학술대회', url: 'https://www.gbcc.kr' },
      { title: '대한암학회', description: '대한암학회 암 진료 권고안', url: 'https://www.cancer.go.kr' },
      { title: '국립암센터', description: '국가 암 관리 및 연구', url: 'https://www.ncc.re.kr' },
      { title: '한국유방건강재단', description: '유방암 예방 및 인식 개선', url: 'https://www.kbcf.or.kr' },
      { title: 'SABCS', description: '샌안토니오 유방암 심포지엄', url: 'https://www.sabcs.org' },
    ]
  }
};

export default function KnowledgeHub() {
  const { user } = useAuth();

  // 기본값을 호흡기내과로 설정하고, user 정보가 로드되면 useEffect에서 업데이트
  const [currentDept, setCurrentDept] = useState<DeptContent>(departmentConfig['호흡기내과']);

  const [searchQuery, setSearchQuery] = useState(currentDept.defaultSearch);
  const [newsQuery, setNewsQuery] = useState(currentDept.defaultNews);
  const [newsType, setNewsType] = useState("all");
  const [literatureData, setLiteratureData] = useState<any>(null);
  const [newsData, setNewsData] = useState<any>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [newsLoading, setNewsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user && user.department && departmentConfig[user.department]) {
      const dept = departmentConfig[user.department];
      setCurrentDept(dept);
      setSearchQuery(dept.defaultSearch);
      setNewsQuery(dept.defaultNews);
    }
  }, [user]);

  // 검색 쿼리가 변경되면(부서 변경 등) 자동으로 검색 실행
  useEffect(() => {
    searchLiterature();
    searchNews();
  }, [currentDept]); # user 변경 시 쿼리 변경 -> currentDept 변경 -> 검색 실행 순서

  const searchLiterature = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const response = await apiRequest("GET", `/api/literature/search?q=${encodeURIComponent(searchQuery)}&max=20`);
      setLiteratureData(response);
    } catch (error) {
      console.error("문헌 검색 오류:", error);
      setError("문헌 검색 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const searchNews = async () => {
    if (!newsQuery.trim()) return;

    setNewsLoading(true);
    try {
      const apiUrl = `/api/literature/news?q=${encodeURIComponent(newsQuery)}&type=${newsType}`;
      console.log("Requesting News API URL:", apiUrl);
      const response = await apiRequest("GET", apiUrl);
      if (response && response.items) {
        const sortedItems = response.items.sort((a: NewsItem, b: NewsItem) => {
          return new Date(b.published).getTime() - new Date(a.published).getTime();
        });
        setNewsData({ ...response, items: sortedItems });
      } else {
        setNewsData(response);
      }
    } catch (error) {
      console.error("뉴스 검색 오류:", error);
    } finally {
      setNewsLoading(false);
    }
  };

  const openPDF = async (url?: string, doi?: string) => {
    if (url) {
      setPdfUrl(url);
      return;
    }

    if (doi) {
      try {
        const response = await apiRequest("GET", `/api/literature/openaccess?doi=${encodeURIComponent(doi)}`);
        if (response.pdf) {
          setPdfUrl(response.pdf);
        } else {
          alert("오픈액세스 PDF를 찾을 수 없습니다.");
        }
      } catch (error) {
        console.error("오픈액세스 검색 오류:", error);
        alert("오픈액세스 검색 중 오류가 발생했습니다.");
      }
    }
  };

  const closePDF = () => {
    setPdfUrl(null);
  };

  useEffect(() => {
    searchLiterature();
    searchNews();
  }, [newsType]); // 뉴스 타입 변경 시 재검색

  const formatDate = (dateString: string) => {
    if (!dateString) return '';
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) {
        return dateString;
      }
      return date.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">{currentDept.title}</h1>
        <p className="text-gray-600 mt-2">
          {currentDept.description}
        </p>
      </div>

      <Tabs defaultValue="papers" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="papers">논문</TabsTrigger>
          <TabsTrigger value="guidelines">가이드라인</TabsTrigger>
          <TabsTrigger value="news">뉴스</TabsTrigger>
        </TabsList>

        {/* 논문 탭 */}
        <TabsContent value="papers" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>논문 검색</CardTitle>
              <CardDescription>
                PubMed과 arXiv에서 최신 논문을 검색하세요.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder={`검색어를 입력하세요 (예: ${currentDept.defaultSearch})`}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1"
                />
                <Button onClick={searchLiterature} disabled={loading}>
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  검색
                </Button>
              </div>
            </CardContent>
          </Card>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {loading ? (
            <div className="flex justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : literatureData ? (
            <div className="space-y-6">
              {/* PubMed 결과 */}
              {literatureData.pubmed?.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5" />
                      PubMed ({literatureData.pubmed.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {literatureData.pubmed.map((paper: PubMedPaper) => (
                        <div key={paper.id} className="border rounded-lg p-4 hover:bg-gray-50">
                          <div className="flex justify-between items-start mb-2">
                            <h3 className="font-semibold text-lg flex-1 mr-4">{paper.title}</h3>
                            <Button variant="outline" size="sm" asChild>
                              <a href={paper.url} target="_blank" rel="noopener noreferrer">
                                <ExternalLink className="h-4 w-4" />
                              </a>
                            </Button>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-600 mb-2">
                            <span className="flex items-center gap-1">
                              <Building className="h-4 w-4" />
                              {paper.journal}
                            </span>
                            <span className="flex items-center gap-1">
                              <Calendar className="h-4 w-4" />
                              {paper.year}
                            </span>
                          </div>
                          <div className="text-sm text-gray-600 mb-3">
                            <span className="flex items-center gap-1">
                              <User className="h-4 w-4" />
                              {paper.authors.slice(0, 5).join(", ")}
                              {paper.authors.length > 5 && " 등"}
                            </span>
                          </div>
                          <div className="flex gap-2">
                            {paper.pmc && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => openPDF(`https://www.ncbi.nlm.nih.gov/pmc/articles/${paper.pmc}/pdf`)}
                              >
                                PDF 열기
                              </Button>
                            )}
                            {paper.doi && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => openPDF(undefined, paper.doi)}
                              >
                                오픈액세스 찾기
                              </Button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* arXiv 결과 */}
              {literatureData.arxiv?.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5" />
                      arXiv ({literatureData.arxiv.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {literatureData.arxiv.map((paper: ArXivPaper) => (
                        <div key={paper.id} className="border rounded-lg p-4 hover:bg-gray-50">
                          <div className="flex justify-between items-start mb-2">
                            <h3 className="font-semibold text-lg flex-1 mr-4">{paper.title}</h3>
                            <Button variant="outline" size="sm" asChild>
                              <a href={paper.url} target="_blank" rel="noopener noreferrer">
                                <ExternalLink className="h-4 w-4" />
                              </a>
                            </Button>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-600 mb-2">
                            <span className="flex items-center gap-1">
                              <Calendar className="h-4 w-4" />
                              {paper.year}
                            </span>
                            <Badge variant="secondary">preprint</Badge>
                          </div>
                          <div className="text-sm text-gray-600 mb-3">
                            <span className="flex items-center gap-1">
                              <User className="h-4 w-4" />
                              {paper.authors.slice(0, 5).join(", ")}
                              {paper.authors.length > 5 && " 등"}
                            </span>
                          </div>
                          <p className="text-sm text-gray-700 mb-3 line-clamp-3">{paper.summary}</p>
                          {paper.pdf && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => openPDF(paper.pdf)}
                            >
                              PDF 열기
                            </Button>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          ) : null}
        </TabsContent>

        {/* 가이드라인 탭 */}
        <TabsContent value="guidelines" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>가이드라인</CardTitle>
              <CardDescription>
                주요 기관의 {currentDept.title.includes('유방암') ? '유방암' : '폐암'} 진료 가이드라인을 확인하세요. 각 링크는 실제 가이드라인 메인 페이지로 연결됩니다.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {currentDept.guidelines.map((guide, index) => (
                  <Card key={index} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardContent className="p-4">
                      <h3 className="font-semibold mb-2">{guide.title}</h3>
                      <p className="text-sm text-gray-600 mb-3">{guide.description}</p>
                      <Button variant="outline" size="sm" className="w-full" asChild>
                        <a href={guide.url} target="_blank" rel="noopener noreferrer">
                          바로가기
                        </a>
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 뉴스 탭 */}
        <TabsContent value="news" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>최신 뉴스</CardTitle>
              <CardDescription>
                {currentDept.title.includes('유방암') ? '유방암' : '폐암'} 관련 최신 의료 뉴스와 연구 동향을 확인하세요.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2 mb-4">
                <Input
                  placeholder={`뉴스 검색어를 입력하세요 (예: ${currentDept.defaultNews})`}
                  value={newsQuery}
                  onChange={(e) => setNewsQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') searchNews(); }}
                  className="flex-1"
                />
                <select
                  value={newsType}
                  onChange={(e) => setNewsType(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">전체</option>
                  <option value="domestic">국내</option>
                  <option value="international">해외</option>
                </select>
                <Button onClick={searchNews} disabled={newsLoading}>
                  {newsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  검색
                </Button>
              </div>
            </CardContent>
          </Card>

          {newsLoading ? (
            <div className="flex justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : newsData?.items?.length > 0 ? (
            <Card>
              <CardContent className="p-6">
                <div className="space-y-4">
                  {newsData.items.map((item: NewsItem, index: number) => (
                    <div key={index} className="border-b pb-4 last:border-b-0">
                      <div className="flex items-start justify-between gap-4">
                        <h3 className="font-semibold mb-2 flex-1">
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:text-blue-800 flex items-center gap-2"
                          >
                            {item.title}
                            <ExternalLink className="h-4 w-4" />
                          </a>
                        </h3>
                        {item.type && (
                          <span
                            className={`text-xs font-medium px-2 py-1 rounded-full ${item.type === 'domestic'
                                ? 'bg-blue-100 text-blue-700'
                                : 'bg-emerald-100 text-emerald-700'
                              }`}
                          >
                            {item.type === 'domestic' ? '국내' : '해외'}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center justify-between text-sm text-gray-500">
                        <div className="flex items-center gap-2">
                          <Globe className="h-3 w-3" />
                          <span>{item.source}</span>
                          <span>{formatDate(item.published)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-12 text-center text-gray-500">
                뉴스 데이터가 없습니다.
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* PDF 뷰어 모달 */}
      {pdfUrl && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg w-full max-w-6xl h-[90vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold">문서 뷰어</h3>
              <Button variant="outline" onClick={closePDF}>
                닫기
              </Button>
            </div>
            <div className="flex-1">
              <iframe
                src={pdfUrl}
                className="w-full h-full"
                title="PDF Viewer"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
