import { useEffect, useState } from "react";
import { Download, Smartphone, Shield, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ReleaseAsset {
  name: string;
  browser_download_url: string;
  size: number;
}

interface GitHubRelease {
  tag_name: string;
  name: string;
  published_at: string;
  body: string;
  assets: ReleaseAsset[];
}

export default function AppDownload() {
  const [release, setRelease] = useState<GitHubRelease | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLatestRelease = async () => {
      try {
        const response = await fetch(
          "https://api.github.com/repos/nogeonu/Flutter/releases/latest"
        );
        if (!response.ok) {
          throw new Error("최신 릴리즈를 가져올 수 없습니다.");
        }
        const data = await response.json();
        setRelease(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
      } finally {
        setLoading(false);
      }
    };

    fetchLatestRelease();
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("ko-KR", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  const apkAsset = release?.assets.find((asset) => asset.name.endsWith(".apk"));

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-12 max-w-5xl">
        {/* 헤더 */}
        <div className="text-center mb-12">
          <Badge className="mb-4 bg-blue-600 hover:bg-blue-700">
            모바일 앱 다운로드
          </Badge>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            건양대학교병원 환자 앱
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            언제 어디서나 간편하게 진료 예약, 건강 기록 확인, 의료진과의 소통이 가능한
            스마트 헬스케어 애플리케이션입니다.
          </p>
        </div>

        {/* 기능 소개 */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <Card className="border-2 hover:border-blue-400 transition-colors">
            <CardHeader>
              <Smartphone className="w-10 h-10 text-blue-600 mb-2" />
              <CardTitle className="text-lg">간편한 예약</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                병원 방문 없이 모바일에서 진료 예약부터 확인까지 한 번에
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-purple-400 transition-colors">
            <CardHeader>
              <Shield className="w-10 h-10 text-purple-600 mb-2" />
              <CardTitle className="text-lg">안전한 개인정보</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                암호화된 통신으로 개인 건강 정보를 안전하게 보호합니다
              </p>
            </CardContent>
          </Card>

          <Card className="border-2 hover:border-green-400 transition-colors">
            <CardHeader>
              <Zap className="w-10 h-10 text-green-600 mb-2" />
              <CardTitle className="text-lg">실시간 알림</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                예약 확인, 진료 일정 등 중요한 정보를 푸시 알림으로 전달
              </p>
            </CardContent>
          </Card>
        </div>

        {/* 다운로드 섹션 */}
        <Card className="shadow-xl border-2">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-lg">
            <CardTitle className="text-2xl flex items-center gap-2">
              <Download className="w-6 h-6" />
              최신 버전 다운로드
            </CardTitle>
            <CardDescription className="text-blue-100">
              Android APK 파일을 다운로드하여 설치하세요
            </CardDescription>
          </CardHeader>
          <CardContent className="p-8">
            {loading && (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <p className="mt-4 text-gray-600">최신 버전 정보를 불러오는 중...</p>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                <p className="text-red-600 font-medium">{error}</p>
                <p className="text-sm text-red-500 mt-2">
                  잠시 후 다시 시도해주세요.
                </p>
              </div>
            )}

            {!loading && !error && release && apkAsset && (
              <div className="space-y-6">
                <div className="bg-gray-50 rounded-lg p-6 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">버전</span>
                    <Badge variant="outline" className="text-base">
                      {release.tag_name}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">파일 크기</span>
                    <span className="text-sm text-gray-900">
                      {formatFileSize(apkAsset.size)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">배포일</span>
                    <span className="text-sm text-gray-900">
                      {formatDate(release.published_at)}
                    </span>
                  </div>
                </div>

                {release.body && (
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">
                      릴리즈 노트
                    </h3>
                    <div className="bg-blue-50 rounded-lg p-4 text-sm text-gray-700 whitespace-pre-wrap">
                      {release.body}
                    </div>
                  </div>
                )}

                <Button
                  size="lg"
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-6 text-lg"
                  onClick={() => window.open(apkAsset.browser_download_url, "_blank")}
                >
                  <Download className="w-5 h-5 mr-2" />
                  APK 다운로드 ({formatFileSize(apkAsset.size)})
                </Button>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-800">
                  <p className="font-medium mb-1">⚠️ 설치 안내</p>
                  <p>
                    APK 파일을 설치하려면 Android 기기의 "출처를 알 수 없는 앱 설치" 권한이
                    필요할 수 있습니다. 설정 → 보안 → 알 수 없는 출처 허용에서 활성화해주세요.
                  </p>
                </div>
              </div>
            )}

            {!loading && !error && (!release || !apkAsset) && (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center">
                <p className="text-gray-600">
                  현재 다운로드 가능한 APK 파일이 없습니다.
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  새로운 버전이 곧 출시될 예정입니다.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 추가 정보 */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>앱 사용 중 문제가 발생하면 병원 고객센터(1234-5678)로 문의해주세요.</p>
          <p className="mt-2">
            GitHub Repository:{" "}
            <a
              href="https://github.com/nogeonu/Flutter"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              nogeonu/Flutter
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

