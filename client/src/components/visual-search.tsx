import { useState, useRef } from "react";
import { Camera, Upload } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { type ProductWithSimilarity } from "@shared/schema";
import { compressImage, dataURLToBase64 } from "@/lib/image-utils";

interface VisualSearchProps {
  onSearchResults: (results: ProductWithSimilarity[]) => void;
  isSearching: boolean;
  setIsSearching: (searching: boolean) => void;
}

export default function VisualSearch({ onSearchResults, isSearching, setIsSearching }: VisualSearchProps) {
  const [recentSearches] = useState([
    "https://images.unsplash.com/photo-1583394838336-acd977736f90?ixlib=rb-4.0.3&auto=format&fit=crop&w=60&h=60",
    "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?ixlib=rb-4.0.3&auto=format&fit=crop&w=60&h=60",
    "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?ixlib=rb-4.0.3&auto=format&fit=crop&w=60&h=60"
  ]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const visualSearchMutation = useMutation({
    mutationFn: async (imageData: string) => {
      const response = await apiRequest("POST", "/api/visual-search", {
        imageData,
        results: ""
      });
      return response.json() as Promise<ProductWithSimilarity[]>;
    },
    onSuccess: (results) => {
      onSearchResults(results);
      setIsSearching(false);
      toast({
        title: "Search Complete",
        description: `Found ${results.length} matching products`,
      });
    },
    onError: () => {
      setIsSearching(false);
      toast({
        title: "Search Failed",
        description: "Unable to process the image. Please try again.",
        variant: "destructive",
      });
    }
  });

  const handleImageCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" }
      });
      
      // Create video element and canvas for capture
      const video = document.createElement("video");
      video.srcObject = stream;
      video.play();
      
      video.addEventListener("loadedmetadata", () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        
        if (ctx) {
          ctx.drawImage(video, 0, 0);
          const imageData = canvas.toDataURL("image/jpeg", 0.8);
          
          // Stop the stream
          stream.getTracks().forEach(track => track.stop());
          
          // Process the captured image
          processImage(imageData);
        }
      });
    } catch (error) {
      toast({
        title: "Camera Access Denied",
        description: "Please allow camera access to take photos.",
        variant: "destructive",
      });
    }
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        processImage(e.target.result as string);
      }
    };
    reader.readAsDataURL(file);
  };

  const processImage = async (imageData: string) => {
    try {
      setIsSearching(true);
      
      // Compress the image for faster upload
      const compressedImage = await compressImage(imageData, 0.7, 800);
      const base64Data = dataURLToBase64(compressedImage);
      
      visualSearchMutation.mutate(base64Data);
    } catch (error) {
      setIsSearching(false);
      toast({
        title: "Image Processing Failed",
        description: "Unable to process the selected image.",
        variant: "destructive",
      });
    }
  };

  return (
    <section className="gradient-hero text-white py-12 px-4">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-3xl sm:text-4xl font-bold mb-4">Find Products Instantly</h2>
        <p className="text-lg mb-8 text-blue-100">Take a photo or upload an image to discover products at the event</p>
        
        {/* Search Interface */}
        <div className="glass-effect rounded-2xl p-6 mb-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Camera Capture Button */}
            <button 
              onClick={handleImageCapture}
              disabled={isSearching}
              className="bg-white text-primary font-semibold py-4 px-6 rounded-xl hover:bg-blue-50 transition-all transform hover:scale-105 flex items-center justify-center space-x-3 disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="button-camera"
            >
              <Camera className="w-5 h-5" />
              <span>Take Photo</span>
            </button>
            
            {/* Upload Button */}
            <button 
              onClick={handleFileUpload}
              disabled={isSearching}
              className="bg-white/20 text-white font-semibold py-4 px-6 rounded-xl hover:bg-white/30 transition-all transform hover:scale-105 flex items-center justify-center space-x-3 border border-white/30 disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="button-upload"
            >
              <Upload className="w-5 h-5" />
              <span>Upload Image</span>
            </button>
          </div>
          
          {/* Hidden File Input */}
          <input 
            ref={fileInputRef}
            type="file" 
            accept="image/*" 
            capture="environment" 
            className="hidden" 
            onChange={handleFileChange}
            data-testid="input-file"
          />

          {/* Loading State */}
          {isSearching && (
            <div className="mt-4 flex items-center justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
              <span className="text-white">Analyzing image...</span>
            </div>
          )}
        </div>

        {/* Recent Searches */}
        <div className="text-left">
          <h3 className="text-sm font-medium text-blue-200 mb-3">Recent Searches</h3>
          <div className="flex space-x-3 overflow-x-auto pb-2">
            {recentSearches.map((search, index) => (
              <div key={index} className="flex-shrink-0 bg-white/15 rounded-lg p-2">
                <img 
                  src={search} 
                  alt="Recent search item" 
                  className="w-12 h-12 rounded object-cover cursor-pointer hover:scale-105 transition-transform"
                  onClick={() => processImage(search)}
                  data-testid={`recent-search-${index}`}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
