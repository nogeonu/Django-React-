import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  ExternalLink,
  FileText,
  Search,
  Loader2,
  Calendar,
  User,
  Building,
  Globe,
  BookOpen,
  ArrowRight,
  Bookmark,
  Share2,
  Filter,
  Download
} from 'lucide-react';
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
    title: 'Pulmonary Research Hub',
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
    title: 'Surgical Oncology Hub',
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

  useEffect(() => {
    searchLiterature();
    searchNews();
  }, [currentDept, newsType]);

  const searchLiterature = async () => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await apiRequest("GET", `/api/literature/search?q=${encodeURIComponent(searchQuery)}&max=20`);
      setLiteratureData(response);
    } catch (error) {
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
      console.error("News search error:", error);
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
        if (response.pdf) setPdfUrl(response.pdf);
        else alert("PDF를 찾을 수 없습니다.");
      } catch (error) {
        alert("오류가 발생했습니다.");
      }
    }
  };

  const closePDF = () => setPdfUrl(null);

  const formatDate = (dateString: string) => {
    if (!dateString) return '';
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return dateString;
      return date.toLocaleDateString('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' });
    } catch { return dateString; }
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.1 } }
  };

  const cardVariants = {
    hidden: { opacity: 0, x: -10 },
    visible: { opacity: 1, x: 0 }
  };

  return (
    <motion.div
      initial="hidden" animate="visible" variants={containerVariants}
      className="space-y-8"
    >
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="bg-blue-600 p-2 rounded-xl shadow-lg shadow-blue-200">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-3xl font-black text-gray-900 tracking-tight">{currentDept.title}</h1>
          </div>
          <p className="text-sm font-medium text-gray-400 max-w-2xl">{currentDept.description}</p>
        </div>
      </div>

      <Tabs defaultValue="papers" className="w-full">
        <TabsList className="bg-white p-1 rounded-2xl shadow-sm border border-gray-100 flex h-14 w-fit mb-8">
          <TabsTrigger value="papers" className="rounded-xl px-8 font-black text-xs uppercase tracking-widest data-[state=active]:bg-blue-600 data-[state=active]:text-white">Publications</TabsTrigger>
          <TabsTrigger value="guidelines" className="rounded-xl px-8 font-black text-xs uppercase tracking-widest data-[state=active]:bg-blue-600 data-[state=active]:text-white">Guidelines</TabsTrigger>
          <TabsTrigger value="news" className="rounded-xl px-8 font-black text-xs uppercase tracking-widest data-[state=active]:bg-blue-600 data-[state=active]:text-white">Medical News</TabsTrigger>
        </TabsList>

        <TabsContent value="papers" className="space-y-8">
          <Card className="border-none shadow-sm rounded-3xl bg-white overflow-hidden">
            <CardHeader className="bg-gray-50/50 border-b border-gray-100 p-8">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                  <CardTitle className="text-xl font-bold text-gray-900 tracking-tight">지식 정보 검색</CardTitle>
                  <CardDescription className="text-xs font-bold text-gray-400 uppercase tracking-widest mt-1">PubMed & arXiv Multi-search</CardDescription>
                </div>
                <div className="flex gap-2 flex-1 md:max-w-xl">
                  <div className="relative flex-1 group">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-blue-600 transition-colors" />
                    <Input
                      placeholder="검색어를 입력하세요..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && searchLiterature()}
                      className="pl-11 h-12 bg-white border-none rounded-2xl ring-1 ring-gray-100 focus-visible:ring-2 focus-visible:ring-blue-600/20"
                    />
                  </div>
                  <Button
                    onClick={searchLiterature}
                    disabled={loading}
                    className="h-12 px-6 rounded-2xl bg-gray-900 hover:bg-black font-black text-xs group"
                  >
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-8">
              {loading ? (
                <div className="flex flex-col items-center justify-center py-20 gap-4">
                  <Loader2 className="h-10 w-10 animate-spin text-blue-600" />
                  <p className="text-xs font-black text-gray-300 uppercase tracking-widest">Searching Databases</p>
                </div>
              ) : literatureData ? (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                  {/* PubMed */}
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="font-black text-xs uppercase tracking-[0.2em] text-blue-600 flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-blue-600 animate-pulse"></div>
                        PubMed Results
                      </h3>
                      <Badge variant="secondary" className="rounded-lg bg-gray-50 text-gray-400 font-bold border-none">{literatureData.pubmed?.length || 0}</Badge>
                    </div>
                    <div className="space-y-4">
                      {literatureData.pubmed?.map((paper: PubMedPaper) => (
                        <motion.div variants={cardVariants} key={paper.id} className="p-6 rounded-3xl border border-gray-50 hover:bg-gray-50/50 hover:border-gray-100 transition-all group relative">
                          <div className="flex justify-between items-start gap-4 mb-3">
                            <h4 className="font-bold text-gray-900 leading-tight group-hover:text-blue-600 transition-colors">{paper.title}</h4>
                            <div className="flex gap-1">
                              <Button variant="ghost" size="icon" className="h-8 w-8 rounded-lg text-gray-300 hover:text-blue-600"><Bookmark className="w-4 h-4" /></Button>
                              <Button variant="ghost" size="icon" className="h-8 w-8 rounded-lg text-gray-300 hover:text-blue-600" asChild>
                                <a href={paper.url} target="_blank" rel="noopener noreferrer"><ExternalLink className="w-4 h-4" /></a>
                              </Button>
                            </div>
                          </div>
                          <div className="flex flex-wrap items-center gap-3 text-[10px] font-black text-gray-400 uppercase tracking-wider mb-4">
                            <span className="flex items-center gap-1"><Building className="w-3 h-3 text-blue-500" /> {paper.journal}</span>
                            <span className="flex items-center gap-1"><Calendar className="w-3 h-3 text-blue-500" /> {paper.year}</span>
                            <span className="flex items-center gap-1"><User className="w-3 h-3 text-blue-500" /> {paper.authors[0]} et al.</span>
                          </div>
                          <div className="flex gap-2">
                            {paper.pmc && (
                              <Button variant="secondary" size="sm" onClick={() => openPDF(`https://www.ncbi.nlm.nih.gov/pmc/articles/${paper.pmc}/pdf`)} className="rounded-xl h-8 text-[10px] font-black bg-blue-50 text-blue-600 hover:bg-blue-600 hover:text-white border-none">
                                <Download className="w-3 h-3 mr-1" /> PDF ACCESS
                              </Button>
                            )}
                            {paper.doi && (
                              <Button variant="outline" size="sm" onClick={() => openPDF(undefined, paper.doi)} className="rounded-xl h-8 text-[10px] font-black border-gray-100 text-gray-400 hover:bg-gray-900 hover:text-white">
                                FIND OPEN ACCESS
                              </Button>
                            )}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>

                  {/* arXiv */}
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="font-black text-xs uppercase tracking-[0.2em] text-purple-600 flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-purple-600 animate-pulse"></div>
                        arXiv preprints
                      </h3>
                      <Badge variant="secondary" className="rounded-lg bg-gray-50 text-gray-400 font-bold border-none">{literatureData.arxiv?.length || 0}</Badge>
                    </div>
                    <div className="space-y-4">
                      {literatureData.arxiv?.map((paper: ArXivPaper) => (
                        <motion.div variants={cardVariants} key={paper.id} className="p-6 rounded-3xl border border-gray-50 hover:bg-gray-50/50 hover:border-gray-100 transition-all group">
                          <h4 className="font-bold text-gray-900 leading-tight mb-3 group-hover:text-purple-600 transition-colors">{paper.title}</h4>
                          <div className="flex flex-wrap items-center gap-3 text-[10px] font-black text-gray-400 uppercase tracking-wider mb-3">
                            <span className="flex items-center gap-1"><Calendar className="w-3 h-3 text-purple-500" /> {paper.year}</span>
                            <Badge className="bg-purple-100 text-purple-600 border-none text-[8px] h-4">PREPRINT</Badge>
                          </div>
                          <p className="text-[11px] font-medium text-gray-400 line-clamp-2 leading-relaxed mb-4">{paper.summary}</p>
                          {paper.pdf && (
                            <Button variant="secondary" size="sm" onClick={() => openPDF(paper.pdf)} className="rounded-xl h-8 text-[10px] font-black bg-purple-50 text-purple-600 hover:bg-purple-600 hover:text-white border-none">
                              <Download className="w-3 h-3 mr-1" /> DOWNLOAD PDF
                            </Button>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="guidelines">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {currentDept.guidelines.map((guide, idx) => (
              <motion.div key={idx} variants={cardVariants}>
                <Card className="border-none shadow-sm rounded-3xl hover:shadow-xl hover:shadow-blue-900/5 transition-all group h-full flex flex-col">
                  <CardContent className="p-8 flex-1 flex flex-col">
                    <div className="bg-blue-50 w-12 h-12 rounded-2xl flex items-center justify-center mb-6 text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                      <FileText className="w-6 h-6" />
                    </div>
                    <h3 className="text-lg font-black text-gray-900 mb-2 leading-tight tracking-tight">{guide.title}</h3>
                    <p className="text-xs font-medium text-gray-400 leading-relaxed mb-8 flex-1">{guide.description}</p>
                    <Button className="w-full h-11 rounded-xl bg-gray-50 text-gray-900 hover:bg-gray-900 hover:text-white font-black text-[10px] tracking-widest uppercase transition-all" asChild>
                      <a href={guide.url} target="_blank" rel="noopener noreferrer">Check Guideline</a>
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="news" className="space-y-8">
          <Card className="border-none shadow-sm rounded-3xl bg-white overflow-hidden">
            <CardHeader className="p-8 pb-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                  <CardTitle className="text-xl font-bold">Medical News Stream</CardTitle>
                  <CardDescription className="text-xs font-bold text-gray-400 uppercase tracking-widest mt-1">Real-time Updates from Medical Journals</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Input
                    placeholder="Search news..."
                    value={newsQuery}
                    onChange={(e) => setNewsQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && searchNews()}
                    className="h-11 rounded-xl bg-gray-50 border-none min-w-[200px]"
                  />
                  <select
                    value={newsType}
                    onChange={(e) => setNewsType(e.target.value)}
                    className="h-11 px-4 bg-gray-50 rounded-xl border-none text-[11px] font-black uppercase outline-none focus:ring-2 focus:ring-blue-600/10 cursor-pointer"
                  >
                    <option value="all">Global</option>
                    <option value="domestic">Domestic</option>
                    <option value="international">Int'l</option>
                  </select>
                  <Button onClick={searchNews} className="h-11 w-11 rounded-xl bg-blue-600 hover:bg-blue-700">
                    <Search className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              {newsLoading ? (
                <div className="py-20 flex flex-col items-center gap-4">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                  <p className="text-[10px] font-black text-gray-300 uppercase tracking-widest">Fetching latest news</p>
                </div>
              ) : (
                <div className="divide-y divide-gray-50">
                  {newsData?.items?.map((item: NewsItem, idx: number) => (
                    <motion.div variants={cardVariants} key={idx} className="p-8 hover:bg-gray-50/50 transition-colors flex flex-col md:flex-row gap-6 items-start group">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-3">
                          {item.type && (
                            <Badge className={`rounded-lg border-none text-[8px] font-black uppercase px-2 py-0.5 h-4 ${item.type === 'domestic' ? 'bg-blue-50 text-blue-600' : 'bg-emerald-50 text-emerald-600'}`}>
                              {item.type}
                            </Badge>
                          )}
                          <span className="text-[10px] font-black text-gray-300 uppercase tracking-widest">{item.source}</span>
                        </div>
                        <h4 className="text-lg font-bold text-gray-900 leading-snug group-hover:text-blue-600 transition-colors mb-2">
                          <a href={item.url} target="_blank" rel="noopener noreferrer">{item.title}</a>
                        </h4>
                        <div className="flex items-center gap-4 text-[10px] font-bold text-gray-400">
                          <span className="flex items-center gap-1"><Calendar className="w-3 h-3" /> {formatDate(item.published)}</span>
                          <span className="flex items-center gap-1 group-hover:text-blue-600 transition-colors cursor-pointer"><Share2 className="w-3 h-3" /> Share</span>
                        </div>
                      </div>
                      <Button variant="ghost" size="icon" className="rounded-xl h-12 w-12 hover:bg-blue-600 hover:text-white transition-colors" asChild>
                        <a href={item.url} target="_blank" rel="noopener noreferrer"><ArrowRight className="w-5 h-5" /></a>
                      </Button>
                    </motion.div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* PDF Mirror */}
      <AnimatePresence>
        {pdfUrl && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-gray-950/80 backdrop-blur-md z-50 flex items-center justify-center p-6 md:p-12"
          >
            <motion.div
              initial={{ scale: 0.95, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.95, y: 20 }}
              className="bg-white rounded-[2.5rem] w-full max-w-7xl h-full flex flex-col shadow-2xl overflow-hidden"
            >
              <div className="flex items-center justify-between p-8 border-b border-gray-100">
                <div className="flex items-center gap-4">
                  <div className="bg-red-50 p-2 rounded-xl"><FileText className="w-5 h-5 text-red-600" /></div>
                  <h3 className="text-xl font-black text-gray-900 tracking-tight">Full Paper Document</h3>
                </div>
                <Button variant="ghost" onClick={closePDF} className="rounded-xl h-11 px-6 font-black text-xs uppercase tracking-widest hover:bg-gray-50">
                  Close Viewer
                </Button>
              </div>
              <div className="flex-1 bg-gray-100">
                <iframe src={pdfUrl} className="w-full h-full border-none" title="PDF Viewer" />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
