import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Home, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-gray-300">404</h1>
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">페이지를 찾을 수 없습니다</h2>
          <p className="text-gray-500 mb-8">
            요청하신 페이지가 존재하지 않거나 이동되었을 수 있습니다.
          </p>
        </div>
        
        <div className="space-x-4">
          <Button asChild>
            <Link to="/">
              <Home className="w-4 h-4 mr-2" />
              홈으로 돌아가기
            </Link>
          </Button>
          <Button variant="outline" onClick={() => window.history.back()}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            이전 페이지
          </Button>
        </div>
      </div>
    </div>
  );
}
