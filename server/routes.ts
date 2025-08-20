import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { 
  insertPatientSchema, 
  insertMedicalImageSchema, 
  insertExaminationSchema,
  insertAiAnalysisResultSchema 
} from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  // Patient routes
  app.get("/api/patients", async (req, res) => {
    try {
      const patients = await storage.getAllPatients();
      res.json(patients);
    } catch (error) {
      res.status(500).json({ message: "환자 목록을 가져오는데 실패했습니다" });
    }
  });

  app.get("/api/patients/:id", async (req, res) => {
    try {
      const patient = await storage.getPatientById(req.params.id);
      if (!patient) {
        return res.status(404).json({ message: "환자를 찾을 수 없습니다" });
      }
      res.json(patient);
    } catch (error) {
      res.status(500).json({ message: "환자 정보를 가져오는데 실패했습니다" });
    }
  });

  app.post("/api/patients", async (req, res) => {
    try {
      const patientData = insertPatientSchema.parse(req.body);
      const patient = await storage.createPatient(patientData);
      res.status(201).json(patient);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ message: "잘못된 데이터입니다", errors: error.errors });
      }
      res.status(500).json({ message: "환자 등록에 실패했습니다" });
    }
  });

  app.put("/api/patients/:id", async (req, res) => {
    try {
      const updateData = req.body;
      const patient = await storage.updatePatient(req.params.id, updateData);
      res.json(patient);
    } catch (error) {
      if (error instanceof Error && error.message === "Patient not found") {
        return res.status(404).json({ message: "환자를 찾을 수 없습니다" });
      }
      res.status(500).json({ message: "환자 정보 수정에 실패했습니다" });
    }
  });

  // Medical Image routes
  app.get("/api/patients/:patientId/images", async (req, res) => {
    try {
      const images = await storage.getMedicalImagesByPatientId(req.params.patientId);
      res.json(images);
    } catch (error) {
      res.status(500).json({ message: "의료 이미지를 가져오는데 실패했습니다" });
    }
  });

  app.post("/api/medical-images", async (req, res) => {
    try {
      const imageData = insertMedicalImageSchema.parse(req.body);
      const image = await storage.uploadMedicalImage(imageData);
      res.status(201).json(image);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ message: "잘못된 데이터입니다", errors: error.errors });
      }
      res.status(500).json({ message: "이미지 업로드에 실패했습니다" });
    }
  });

  // Examination routes
  app.get("/api/patients/:patientId/examinations", async (req, res) => {
    try {
      const examinations = await storage.getExaminationsByPatientId(req.params.patientId);
      res.json(examinations);
    } catch (error) {
      res.status(500).json({ message: "검사 기록을 가져오는데 실패했습니다" });
    }
  });

  app.post("/api/examinations", async (req, res) => {
    try {
      const examData = insertExaminationSchema.parse(req.body);
      const examination = await storage.createExamination(examData);
      res.status(201).json(examination);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ message: "잘못된 데이터입니다", errors: error.errors });
      }
      res.status(500).json({ message: "검사 등록에 실패했습니다" });
    }
  });

  app.put("/api/examinations/:id/status", async (req, res) => {
    try {
      const { status } = req.body;
      const examination = await storage.updateExaminationStatus(req.params.id, status);
      res.json(examination);
    } catch (error) {
      if (error instanceof Error && error.message === "Examination not found") {
        return res.status(404).json({ message: "검사를 찾을 수 없습니다" });
      }
      res.status(500).json({ message: "검사 상태 업데이트에 실패했습니다" });
    }
  });

  // AI Analysis routes
  app.post("/api/ai-analysis", async (req, res) => {
    try {
      const { imageId, imageData } = req.body;
      
      if (!imageId || !imageData) {
        return res.status(400).json({ message: "이미지 ID와 데이터가 필요합니다" });
      }
      
      const result = await storage.performAiAnalysis(imageId, imageData);
      res.json(result);
    } catch (error) {
      res.status(500).json({ message: "AI 분석에 실패했습니다" });
    }
  });

  app.get("/api/images/:imageId/analysis", async (req, res) => {
    try {
      const results = await storage.getAnalysisResultsByImageId(req.params.imageId);
      res.json(results);
    } catch (error) {
      res.status(500).json({ message: "분석 결과를 가져오는데 실패했습니다" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
