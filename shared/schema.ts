import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, timestamp, date } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// 환자 정보 테이블
export const patients = pgTable("patients", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientNumber: varchar("patient_number").notNull().unique(), // 환자번호
  name: text("name").notNull(), // 환자명
  birthDate: date("birth_date").notNull(), // 생년월일
  gender: varchar("gender", { length: 10 }).notNull(), // 성별
  phone: varchar("phone", { length: 20 }), // 전화번호
  email: varchar("email"), // 이메일
  address: text("address"), // 주소
  emergencyContact: text("emergency_contact"), // 응급연락처
  bloodType: varchar("blood_type", { length: 5 }), // 혈액형
  allergies: text("allergies"), // 알레르기
  medicalHistory: text("medical_history"), // 병력
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// 의료 이미지 테이블
export const medicalImages = pgTable("medical_images", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => patients.id),
  examId: varchar("exam_id"), // 검사 ID (나중에 examinations 테이블과 연결)
  imageType: varchar("image_type").notNull(), // MRI, CT, X-RAY 등
  bodyPart: varchar("body_part").notNull(), // 촬영 부위
  imageUrl: text("image_url").notNull(), // 이미지 URL
  originalFileName: text("original_file_name"), // 원본 파일명
  fileSize: integer("file_size"), // 파일 크기 (bytes)
  uploadedAt: timestamp("uploaded_at").defaultNow(),
  description: text("description"), // 설명
});

// 검사 정보 테이블
export const examinations = pgTable("examinations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => patients.id),
  examType: varchar("exam_type").notNull(), // MRI, CT, X-RAY 등
  examDate: timestamp("exam_date").defaultNow(),
  bodyPart: varchar("body_part").notNull(), // 검사 부위
  status: varchar("status").notNull().default("pending"), // pending, in_progress, completed
  doctorName: varchar("doctor_name"), // 담당 의사
  notes: text("notes"), // 검사 메모
  createdAt: timestamp("created_at").defaultNow(),
});

// AI 분석 결과 테이블
export const aiAnalysisResults = pgTable("ai_analysis_results", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  imageId: varchar("image_id").notNull().references(() => medicalImages.id),
  analysisType: varchar("analysis_type").notNull(), // YOLO, classification 등
  results: text("results").notNull(), // JSON 형태의 분석 결과
  confidence: integer("confidence"), // 신뢰도 (0-100)
  findings: text("findings"), // 발견사항
  recommendations: text("recommendations"), // 권장사항
  analysisDate: timestamp("analysis_date").defaultNow(),
  modelVersion: varchar("model_version"), // 사용된 모델 버전
});

// Insert 스키마들
export const insertPatientSchema = createInsertSchema(patients).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertMedicalImageSchema = createInsertSchema(medicalImages).omit({
  id: true,
  uploadedAt: true,
});

export const insertExaminationSchema = createInsertSchema(examinations).omit({
  id: true,
  createdAt: true,
});

export const insertAiAnalysisResultSchema = createInsertSchema(aiAnalysisResults).omit({
  id: true,
  analysisDate: true,
});

// Types
export type Patient = typeof patients.$inferSelect;
export type InsertPatient = z.infer<typeof insertPatientSchema>;
export type MedicalImage = typeof medicalImages.$inferSelect;
export type InsertMedicalImage = z.infer<typeof insertMedicalImageSchema>;
export type Examination = typeof examinations.$inferSelect;
export type InsertExamination = z.infer<typeof insertExaminationSchema>;
export type AiAnalysisResult = typeof aiAnalysisResults.$inferSelect;
export type InsertAiAnalysisResult = z.infer<typeof insertAiAnalysisResultSchema>;

// Extended types for frontend
export type PatientWithExams = Patient & {
  examinations: Examination[];
};

export type ImageWithAnalysis = MedicalImage & {
  analysis?: AiAnalysisResult;
};

export type ExamWithImages = Examination & {
  images: ImageWithAnalysis[];
};
