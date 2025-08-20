import { 
  type Patient, 
  type InsertPatient, 
  type MedicalImage, 
  type InsertMedicalImage, 
  type Examination, 
  type InsertExamination,
  type AiAnalysisResult,
  type InsertAiAnalysisResult,
  type PatientWithExams,
  type ImageWithAnalysis,
  type ExamWithImages
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Patients
  getAllPatients(): Promise<Patient[]>;
  getPatientById(id: string): Promise<Patient | undefined>;
  getPatientByNumber(patientNumber: string): Promise<Patient | undefined>;
  createPatient(patient: InsertPatient): Promise<Patient>;
  updatePatient(id: string, patient: Partial<InsertPatient>): Promise<Patient>;
  
  // Medical Images
  getMedicalImagesByPatientId(patientId: string): Promise<ImageWithAnalysis[]>;
  getMedicalImageById(id: string): Promise<MedicalImage | undefined>;
  uploadMedicalImage(image: InsertMedicalImage): Promise<MedicalImage>;
  
  // Examinations
  getExaminationsByPatientId(patientId: string): Promise<ExamWithImages[]>;
  getExaminationById(id: string): Promise<Examination | undefined>;
  createExamination(exam: InsertExamination): Promise<Examination>;
  updateExaminationStatus(id: string, status: string): Promise<Examination>;
  
  // AI Analysis
  performAiAnalysis(imageId: string, imageData: string): Promise<AiAnalysisResult>;
  getAnalysisResultsByImageId(imageId: string): Promise<AiAnalysisResult[]>;
  saveAnalysisResult(result: InsertAiAnalysisResult): Promise<AiAnalysisResult>;
}

export class MemStorage implements IStorage {
  private patients: Map<string, Patient>;
  private medicalImages: Map<string, MedicalImage>;
  private examinations: Map<string, Examination>;
  private aiAnalysisResults: Map<string, AiAnalysisResult>;

  constructor() {
    this.patients = new Map();
    this.medicalImages = new Map();
    this.examinations = new Map();
    this.aiAnalysisResults = new Map();
    this.initializeSampleData();
  }

  private initializeSampleData() {
    // 샘플 환자 데이터
    const samplePatients: InsertPatient[] = [
      {
        patientNumber: "P2024001",
        name: "김철수",
        birthDate: "1985-03-15",
        gender: "남성",
        phone: "010-1234-5678",
        email: "kim@example.com",
        address: "서울시 강남구 역삼동",
        bloodType: "A+",
        allergies: "페니실린",
        medicalHistory: "고혈압, 당뇨병"
      },
      {
        patientNumber: "P2024002",
        name: "이영희",
        birthDate: "1990-07-22",
        gender: "여성",
        phone: "010-9876-5432",
        email: "lee@example.com",
        address: "서울시 서초구 서초동",
        bloodType: "B+",
        allergies: "없음",
        medicalHistory: "없음"
      },
      {
        patientNumber: "P2024003",
        name: "박민수",
        birthDate: "1978-11-08",
        gender: "남성",
        phone: "010-5555-1234",
        email: "park@example.com",
        address: "서울시 송파구 잠실동",
        bloodType: "O+",
        allergies: "갑각류",
        medicalHistory: "천식"
      }
    ];

    samplePatients.forEach(async (patient) => {
      await this.createPatient(patient);
    });
  }

  // Patient methods
  async getAllPatients(): Promise<Patient[]> {
    return Array.from(this.patients.values());
  }

  async getPatientById(id: string): Promise<Patient | undefined> {
    return this.patients.get(id);
  }

  async getPatientByNumber(patientNumber: string): Promise<Patient | undefined> {
    return Array.from(this.patients.values()).find(p => p.patientNumber === patientNumber);
  }

  async createPatient(insertPatient: InsertPatient): Promise<Patient> {
    const id = randomUUID();
    const patient: Patient = {
      ...insertPatient,
      id,
      createdAt: new Date(),
      updatedAt: new Date(),
      address: insertPatient.address || null,
      phone: insertPatient.phone || null,
      email: insertPatient.email || null,
      emergencyContact: insertPatient.emergencyContact || null,
      bloodType: insertPatient.bloodType || null,
      allergies: insertPatient.allergies || null,
      medicalHistory: insertPatient.medicalHistory || null,
    };
    this.patients.set(id, patient);
    return patient;
  }

  async updatePatient(id: string, updateData: Partial<InsertPatient>): Promise<Patient> {
    const existingPatient = this.patients.get(id);
    if (!existingPatient) {
      throw new Error("Patient not found");
    }
    
    const updatedPatient: Patient = {
      ...existingPatient,
      ...updateData,
      updatedAt: new Date(),
    };
    this.patients.set(id, updatedPatient);
    return updatedPatient;
  }

  // Medical Image methods
  async getMedicalImagesByPatientId(patientId: string): Promise<ImageWithAnalysis[]> {
    const images = Array.from(this.medicalImages.values())
      .filter(img => img.patientId === patientId);
    
    const imagesWithAnalysis = await Promise.all(images.map(async (image) => {
      const analysisResults = await this.getAnalysisResultsByImageId(image.id);
      return {
        ...image,
        analysis: analysisResults[0] // 가장 최근 분석 결과
      };
    }));
    
    return imagesWithAnalysis;
  }

  async getMedicalImageById(id: string): Promise<MedicalImage | undefined> {
    return this.medicalImages.get(id);
  }

  async uploadMedicalImage(insertImage: InsertMedicalImage): Promise<MedicalImage> {
    const id = randomUUID();
    const image: MedicalImage = {
      ...insertImage,
      id,
      uploadedAt: new Date(),
      examId: insertImage.examId || null,
      description: insertImage.description || null,
      originalFileName: insertImage.originalFileName || null,
      fileSize: insertImage.fileSize || null,
    };
    this.medicalImages.set(id, image);
    return image;
  }

  // Examination methods
  async getExaminationsByPatientId(patientId: string): Promise<ExamWithImages[]> {
    const exams = Array.from(this.examinations.values())
      .filter(exam => exam.patientId === patientId);
    
    const examsWithImages = await Promise.all(exams.map(async (exam) => {
      const images = await this.getMedicalImagesByPatientId(patientId);
      const examImages = images.filter(img => img.examId === exam.id);
      return {
        ...exam,
        images: examImages
      };
    }));
    
    return examsWithImages;
  }

  async getExaminationById(id: string): Promise<Examination | undefined> {
    return this.examinations.get(id);
  }

  async createExamination(insertExam: InsertExamination): Promise<Examination> {
    const id = randomUUID();
    const exam: Examination = {
      ...insertExam,
      id,
      createdAt: new Date(),
      status: insertExam.status || "pending",
      examDate: insertExam.examDate || new Date(),
      doctorName: insertExam.doctorName || null,
      notes: insertExam.notes || null,
    };
    this.examinations.set(id, exam);
    return exam;
  }

  async updateExaminationStatus(id: string, status: string): Promise<Examination> {
    const exam = this.examinations.get(id);
    if (!exam) {
      throw new Error("Examination not found");
    }
    
    const updatedExam = { ...exam, status };
    this.examinations.set(id, updatedExam);
    return updatedExam;
  }

  // AI Analysis methods
  async performAiAnalysis(imageId: string, imageData: string): Promise<AiAnalysisResult> {
    // 실제 AI 모델이 없으므로 모의 분석 결과 생성
    const mockFindings = [
      "정상 소견",
      "경미한 이상 소견 발견",
      "추가 검사 필요",
      "즉시 의사와 상담 필요"
    ];
    
    const mockResults = {
      detected_objects: [
        { class: "normal_tissue", confidence: 0.85, bbox: [100, 100, 200, 200] },
        { class: "anomaly", confidence: 0.65, bbox: [150, 150, 250, 250] }
      ],
      classification: "normal",
      risk_score: Math.floor(Math.random() * 100)
    };
    
    const analysisResult: InsertAiAnalysisResult = {
      imageId,
      analysisType: "YOLO_v8",
      results: JSON.stringify(mockResults),
      confidence: Math.floor(Math.random() * 40) + 60, // 60-100%
      findings: mockFindings[Math.floor(Math.random() * mockFindings.length)],
      recommendations: "정기적인 추적 검사를 권장합니다.",
      modelVersion: "v1.0.0"
    };
    
    return await this.saveAnalysisResult(analysisResult);
  }

  async getAnalysisResultsByImageId(imageId: string): Promise<AiAnalysisResult[]> {
    return Array.from(this.aiAnalysisResults.values())
      .filter(result => result.imageId === imageId)
      .sort((a, b) => (b.analysisDate?.getTime() || 0) - (a.analysisDate?.getTime() || 0));
  }

  async saveAnalysisResult(insertResult: InsertAiAnalysisResult): Promise<AiAnalysisResult> {
    const id = randomUUID();
    const result: AiAnalysisResult = {
      ...insertResult,
      id,
      analysisDate: new Date(),
      confidence: insertResult.confidence || null,
      findings: insertResult.findings || null,
      recommendations: insertResult.recommendations || null,
      modelVersion: insertResult.modelVersion || null,
    };
    this.aiAnalysisResults.set(id, result);
    return result;
  }
}

export const storage = new MemStorage();
