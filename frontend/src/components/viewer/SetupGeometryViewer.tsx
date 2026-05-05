import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import {
  fetchHistorySetupField,
  fetchHistorySetupGeometry,
  fetchHistorySnapshots,
  fetchSetupField,
  fetchSetupGeometry,
  fetchSnapshots,
  type ViewerSetupFieldResponse,
  type ViewerSetupGeometryResponse,
} from "../../api";
import { SnapshotTimeline } from "./SnapshotTimeline";

type SetupGeometryViewerProps = {
  source: {
    kind: "live";
    runId: string | null;
    runStatus: string | null;
  } | {
    kind: "history";
    runPath: string | null;
  };
};

type SetupPropertyMode = "material" | "Ui" | "ni" | "Ci" | "ΔUi";
const SETUP_PROPERTY_OPTIONS: SetupPropertyMode[] = ["material", "Ui", "ni", "Ci", "ΔUi"];
type CameraPreset = "iso" | "x" | "y" | "z";

const MATERIAL_COLORS: Record<string, string> = {
  gate: "#eed605",
  backgate: "#eed605",
  top_gate: "#eed605",
  dielectric: "#87d5d8",
  Qsystem: "#d64b1d",
  vacuum: "#ecfafb",
  dopants: "#c897ce",
  unknown: "#b5becb",
};

const VIRIDIS_STOPS = [
  { t: 0, color: "#440154" },
  { t: 0.25, color: "#3b528b" },
  { t: 0.5, color: "#21918c" },
  { t: 0.75, color: "#5ec962" },
  { t: 1, color: "#fde725" },
];
const viridisGradient = `linear-gradient(0deg, ${VIRIDIS_STOPS
  .map((stop) => `${stop.color} ${stop.t * 100}%`)
  .join(", ")})`;

function isInspectable(runStatus: string | null) {
  return runStatus !== null && !["queued", "failed", "aborted"].includes(runStatus);
}

function isPendingStaticError(message: string) {
  return message.includes("Missing run_static.npz");
}

function colorForMaterial(material: string) {
  return MATERIAL_COLORS[material] ?? MATERIAL_COLORS.unknown;
}

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

function colorForFieldValue(value: number | null, low: number, high: number) {
  if (value === null || Number.isNaN(value)) {
    return new THREE.Color("#b5becb");
  }

  const span = high - low || 1;
  const t = clamp((value - low) / span, 0, 1);
  const nextIndex = VIRIDIS_STOPS.findIndex((stop) => t <= stop.t);
  if (nextIndex <= 0) {
    return new THREE.Color(VIRIDIS_STOPS[0].color);
  }
  if (nextIndex === -1) {
    return new THREE.Color(VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1].color);
  }

  const left = VIRIDIS_STOPS[nextIndex - 1];
  const right = VIRIDIS_STOPS[nextIndex];
  const localT = (t - left.t) / (right.t - left.t || 1);
  return new THREE.Color(left.color).lerp(new THREE.Color(right.color), localT);
}

function buildLegend(materials: string[]) {
  return [...new Set(materials)].sort((left, right) => left.localeCompare(right));
}

function createPointSpriteTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }

  context.clearRect(0, 0, size, size);
  const gradient = context.createRadialGradient(size / 2, size / 2, size * 0.12, size / 2, size / 2, size * 0.5);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.7, "rgba(255,255,255,0.96)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.beginPath();
  context.arc(size / 2, size / 2, size * 0.32, 0, Math.PI * 2);
  context.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function presetAngles(preset: CameraPreset) {
  if (preset === "x") {
    return { azimuth: 0, polar: Math.PI / 2 };
  }
  if (preset === "y") {
    return { azimuth: Math.PI / 2, polar: Math.PI / 2 };
  }
  if (preset === "z") {
    return { azimuth: 0, polar: 0.18 };
  }
  return { azimuth: -0.98, polar: 1.05 };
}

export function SetupGeometryViewer({ source }: SetupGeometryViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const snapCameraRef = useRef<((preset: CameraPreset) => void) | null>(null);
  const [geometry, setGeometry] = useState<ViewerSetupGeometryResponse | null>(null);
  const [snapshots, setSnapshots] = useState<string[]>([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState("");
  const [propertyMode, setPropertyMode] = useState<SetupPropertyMode>("material");
  const [fieldData, setFieldData] = useState<ViewerSetupFieldResponse | null>(null);
  const [visibleMaterials, setVisibleMaterials] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingField, setIsLoadingField] = useState(false);

  useEffect(() => {
    if (source.kind === "live" && (!source.runId || !isInspectable(source.runStatus))) {
      setGeometry(null);
      return;
    }
    if (source.kind === "history" && !source.runPath) {
      setGeometry(null);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    const request =
      source.kind === "live"
        ? fetchSetupGeometry(source.runId as string)
        : fetchHistorySetupGeometry(source.runPath as string);

    request
      .then((response) => {
        if (cancelled) {
          return;
        }
        setGeometry(response);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : "Failed to load setup geometry.";
        setGeometry(null);
        if (isPendingStaticError(message)) {
          setError(null);
          return;
        }
        setError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [source.kind, source.kind === "live" ? source.runId : source.runPath, source.kind === "live" ? source.runStatus : null]);

  useEffect(() => {
    if (source.kind === "live" && (!source.runId || !isInspectable(source.runStatus))) {
      setSnapshots([]);
      setSelectedSnapshot("");
      return;
    }
    if (source.kind === "history" && !source.runPath) {
      setSnapshots([]);
      setSelectedSnapshot("");
      return;
    }

    let cancelled = false;

    async function loadSnapshots() {
      try {
        const response =
          source.kind === "live"
            ? await fetchSnapshots(source.runId as string)
            : await fetchHistorySnapshots(source.runPath as string);
        if (cancelled) {
          return;
        }
        setSnapshots(response.snapshots);
        setSelectedSnapshot((current) => {
          if (current && response.snapshots.includes(current)) {
            return current;
          }
          return response.snapshots[response.snapshots.length - 1] ?? "";
        });
      } catch {
        if (!cancelled) {
          setSnapshots([]);
          setSelectedSnapshot("");
        }
      }
    }

    void loadSnapshots();
    const timer = window.setInterval(() => {
      void loadSnapshots();
    }, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [source.kind, source.kind === "live" ? source.runId : source.runPath, source.kind === "live" ? source.runStatus : null]);

  useEffect(() => {
    if (propertyMode === "material" || !selectedSnapshot) {
      setFieldData(null);
      return;
    }
    if (source.kind === "live" && !source.runId) {
      setFieldData(null);
      return;
    }
    if (source.kind === "history" && !source.runPath) {
      setFieldData(null);
      return;
    }

    let cancelled = false;
    setIsLoadingField(true);

    const request =
      source.kind === "live"
        ? fetchSetupField(source.runId as string, selectedSnapshot, propertyMode)
        : fetchHistorySetupField(source.runPath as string, selectedSnapshot, propertyMode);

    request
      .then((response) => {
        if (cancelled) {
          return;
        }
        setFieldData(response);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setFieldData(null);
        setError(err instanceof Error ? err.message : "Failed to load setup field data.");
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingField(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [propertyMode, selectedSnapshot, source.kind, source.kind === "live" ? source.runId : source.runPath]);

  const legend = useMemo(() => buildLegend(geometry?.materials ?? []), [geometry?.materials]);
  const siteCount = geometry?.site_ids.length ?? 0;

  useEffect(() => {
    if (legend.length === 0) {
      setVisibleMaterials([]);
      return;
    }
    setVisibleMaterials((current) => {
      const next = current.filter((material) => legend.includes(material));
      return next.length > 0 ? next : legend;
    });
  }, [legend]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container || !geometry) {
      return;
    }

    const validPoints = geometry.coordinates
      .map((point, index) => ({ point, material: geometry.materials[index] ?? "unknown", siteId: geometry.site_ids[index] }))
      .filter(
        (entry): entry is { point: [number, number, number]; material: string; siteId: number } =>
          entry.point.length >= 3 &&
          typeof entry.point[0] === "number" &&
          typeof entry.point[1] === "number" &&
          typeof entry.point[2] === "number",
      );

    if (validPoints.length === 0) {
      return;
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.replaceChildren(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#fbfbfb");
    const pointSpriteTexture = createPointSpriteTexture();

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.01, 100);
    camera.position.set(2.8, -4.2, 2.8);
    camera.up.set(0, 0, 1);

    const fieldValueBySite = new Map<number, number | null>();
    if (fieldData && propertyMode !== "material") {
      fieldData.site_ids.forEach((siteId, index) => {
        fieldValueBySite.set(siteId, fieldData.values[index] ?? null);
      });
    }

    const root = new THREE.Group();
    scene.add(root);
    if (propertyMode === "material" || !fieldData) {
      const grouped = new Map<string, Array<[number, number, number]>>();
      validPoints.forEach(({ point, material }) => {
        if (visibleMaterials.length > 0 && !visibleMaterials.includes(material)) {
          return;
        }
        const bucket = grouped.get(material) ?? [];
        bucket.push(point);
        grouped.set(material, bucket);
      });

      grouped.forEach((points, material) => {
        const positions = new Float32Array(points.length * 3);
        points.forEach((point, index) => {
          positions[index * 3] = point[0];
          positions[index * 3 + 1] = point[1];
          positions[index * 3 + 2] = point[2];
        });

        const pointGeometry = new THREE.BufferGeometry();
        pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const pointMaterial = new THREE.PointsMaterial({
          color: colorForMaterial(material),
          size: material === "Qsystem" ? 0.038 : 0.03,
          sizeAttenuation: true,
          map: pointSpriteTexture,
          alphaMap: pointSpriteTexture,
          transparent: true,
          alphaTest: 0.2,
          opacity: material === "vacuum" ? 0.8 : material === "dielectric" ? 0.82 : material === "dopants" ? 0.48 : 0.72,
        });
        root.add(new THREE.Points(pointGeometry, pointMaterial));
      });
    } else {
      const filteredPoints = validPoints.filter(
        ({ material }) => visibleMaterials.length === 0 || visibleMaterials.includes(material),
      );
      const positions = new Float32Array(filteredPoints.length * 3);
      const colors = new Float32Array(filteredPoints.length * 3);
      filteredPoints.forEach(({ point, siteId }, index) => {
        positions[index * 3] = point[0];
        positions[index * 3 + 1] = point[1];
        positions[index * 3 + 2] = point[2];

        const color = colorForFieldValue(fieldValueBySite.get(siteId) ?? null, fieldData.color_min, fieldData.color_max);
        colors[index * 3] = color.r;
        colors[index * 3 + 1] = color.g;
        colors[index * 3 + 2] = color.b;
      });

      const pointGeometry = new THREE.BufferGeometry();
      pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      pointGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const pointMaterial = new THREE.PointsMaterial({
        size: 0.036,
        sizeAttenuation: true,
        vertexColors: true,
        map: pointSpriteTexture,
        alphaMap: pointSpriteTexture,
        transparent: true,
        alphaTest: 0.2,
        opacity: 0.78,
      });
      root.add(new THREE.Points(pointGeometry, pointMaterial));
    }

    if (geometry.bounds_min.every((value) => typeof value === "number") && geometry.bounds_max.every((value) => typeof value === "number")) {
      const [minX, minY, minZ] = geometry.bounds_min as [number, number, number];
      const [maxX, maxY, maxZ] = geometry.bounds_max as [number, number, number];
      const boxGeometry = new THREE.BoxGeometry(maxX - minX, maxY - minY, maxZ - minZ);
      const boxMaterial = new THREE.MeshBasicMaterial({
        color: "#8d99a5",
        transparent: true,
        opacity: 0.03,
        wireframe: true,
      });
      const boundsMesh = new THREE.Mesh(boxGeometry, boxMaterial);
      boundsMesh.position.set((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
      root.add(boundsMesh);
    }

    const bounds = new THREE.Box3().setFromObject(root);
    const center = bounds.getCenter(new THREE.Vector3());
    root.position.sub(center);

    const size = bounds.getSize(new THREE.Vector3());
    const gridSize = Math.max(size.x, size.y) * 2.1 || 1;
    const grid = new THREE.GridHelper(gridSize, 12, "#9aa4af", "#cfd5da");
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -size.z * 0.52;
    scene.add(grid);

    const axes = new THREE.AxesHelper(Math.max(size.x, size.y, size.z) * 0.45 || 0.5);
    axes.position.set(-gridSize * 0.38, -gridSize * 0.38, -size.z * 0.52);
    scene.add(axes);

    let isDragging = false;
    let startX = 0;
    let startY = 0;

    const onPointerDown = (event: PointerEvent) => {
      isDragging = true;
      startX = event.clientX;
      startY = event.clientY;
      container.setPointerCapture(event.pointerId);
    };

    const onPointerMove = (event: PointerEvent) => {
      if (!isDragging) {
        return;
      }
      const deltaX = event.clientX - startX;
      const deltaY = event.clientY - startY;
      startX = event.clientX;
      startY = event.clientY;
      azimuth -= deltaX * 0.008;
      polar = Math.max(0.08, Math.min(Math.PI - 0.08, polar + deltaY * 0.008));
      applyCamera();
    };

    const onPointerUp = (event: PointerEvent) => {
      isDragging = false;
      if (container.hasPointerCapture(event.pointerId)) {
        container.releasePointerCapture(event.pointerId);
      }
    };

    let radius = Math.max(size.x, size.y, size.z) * 2.4 || 4;
    let azimuth = -0.98;
    let polar = 1.05;

    const applyCamera = () => {
      camera.position.set(
        radius * Math.sin(polar) * Math.cos(azimuth),
        radius * Math.sin(polar) * Math.sin(azimuth),
        radius * Math.cos(polar),
      );
      camera.lookAt(0, 0, 0);
    };
    applyCamera();
    snapCameraRef.current = (preset: CameraPreset) => {
      const next = presetAngles(preset);
      azimuth = next.azimuth;
      polar = next.polar;
      applyCamera();
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      radius = Math.max(0.5, Math.min(20, radius * (event.deltaY > 0 ? 1.08 : 0.92)));
      applyCamera();
    };

    const onResize = () => {
      const nextWidth = container.clientWidth;
      const nextHeight = container.clientHeight;
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    };

    let frameId = 0;
    const animate = () => {
      frameId = window.requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    container.addEventListener("pointerdown", onPointerDown);
    container.addEventListener("pointermove", onPointerMove);
    container.addEventListener("pointerup", onPointerUp);
    container.addEventListener("pointerleave", onPointerUp);
    container.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("resize", onResize);

    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener("resize", onResize);
      container.removeEventListener("pointerdown", onPointerDown);
      container.removeEventListener("pointermove", onPointerMove);
      container.removeEventListener("pointerup", onPointerUp);
      container.removeEventListener("pointerleave", onPointerUp);
      container.removeEventListener("wheel", onWheel);
      snapCameraRef.current = null;
      renderer.dispose();
      scene.traverse((object: THREE.Object3D) => {
        const mesh = object as THREE.Points<THREE.BufferGeometry, THREE.Material | THREE.Material[]>;
        mesh.geometry?.dispose?.();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach((material: THREE.Material) => material.dispose());
        } else {
          mesh.material?.dispose?.();
        }
      });
      pointSpriteTexture?.dispose();
    };
  }, [fieldData, geometry, propertyMode, visibleMaterials]);

  function toggleMaterial(material: string) {
    setVisibleMaterials((current) =>
      current.includes(material) ? current.filter((item) => item !== material) : [...current, material],
    );
  }

  function showAllMaterials() {
    setVisibleMaterials(legend);
  }

  if ((source.kind === "live" && !source.runId) || (source.kind === "history" && !source.runPath)) {
    return (
      <div className="setup-viewer-empty">
        <p>Start a run to build the geometry, then the interactive setup viewer will render the site cloud here.</p>
      </div>
    );
  }

  if (isLoading && !geometry) {
    return <div className="setup-viewer-empty">Loading setup geometry…</div>;
  }

  if (error && !geometry) {
    return <div className="setup-viewer-empty">{error}</div>;
  }

  if (!geometry) {
    return (
      <div className="setup-viewer-empty">
        <p>Geometry data is not available yet. Once `run_static.npz` is written, this panel will render the setup with Three.js.</p>
      </div>
    );
  }

  return (
    <div className="setup-viewer-shell">
      <div ref={mountRef} className="setup-viewer-canvas" />

      <div className="setup-viewer-gizmo" aria-label="Camera orientation">
        <button type="button" className="gizmo-button gizmo-z" onClick={() => snapCameraRef.current?.("z")}>
          Z
        </button>
        <button type="button" className="gizmo-button gizmo-x" onClick={() => snapCameraRef.current?.("x")}>
          X
        </button>
        <button type="button" className="gizmo-button gizmo-y" onClick={() => snapCameraRef.current?.("y")}>
          Y
        </button>
        <button type="button" className="gizmo-button gizmo-home" onClick={() => snapCameraRef.current?.("iso")}>
          +
        </button>
      </div>

      <div className="setup-viewer-toolbar">
        <div className="display-data-panel">
          <h3 className="panel-title">Display</h3>
          <div className="property-pills">
            {SETUP_PROPERTY_OPTIONS.map((item) => (
              <button
                key={item}
                type="button"
                className={`property-pill ${propertyMode === item ? "active" : ""}`}
                onClick={() => setPropertyMode(item)}
              >
                {item === "material" ? "Material" : item}
              </button>
            ))}
          </div>
        </div>
      </div>

      {snapshots.length > 0 ? (
        <div className="timeline-overlay">
          <SnapshotTimeline
            snapshots={snapshots}
            activeSnapshot={selectedSnapshot}
            onSelect={setSelectedSnapshot}
          />
        </div>
      ) : propertyMode !== "material" ? (
        <div className="timeline-overlay setup-timeline-empty">Waiting for snapshots…</div>
      ) : null}

      <div className="setup-material-window">
        <div className="setup-material-filter">
          <button
            type="button"
            className={`setup-material-pill ${visibleMaterials.length === legend.length ? "active" : ""}`}
            onClick={showAllMaterials}
          >
            all
          </button>
          {legend.map((material) => (
            <button
              key={material}
              type="button"
              className={`setup-material-pill ${visibleMaterials.includes(material) ? "active" : ""}`}
              onClick={() => toggleMaterial(material)}
            >
              <span className="legend-swatch" style={{ background: colorForMaterial(material) }} />
              {material}
            </button>
          ))}
        </div>
      </div>

      <div className="setup-legend">
        {propertyMode !== "material" && fieldData ? (
          <>
            <div className="setup-legend-range">
              <span className="setup-legend-max">{fieldData.color_max.toPrecision(4)}</span>
              <div className="setup-legend-bar" style={{ background: viridisGradient }} />
              <span className="setup-legend-min">{fieldData.color_min.toPrecision(4)}</span>
            </div>
            <div className="setup-legend-caption">{fieldData.property}</div>
          </>
        ) : null}
      </div>

      <div className="setup-footnote">
        <div>
          <span>Points</span>
          <strong>{siteCount}</strong>
        </div>
        <div>
          <span>Quantum sites</span>
          <strong>{geometry.qsite_ids.length}</strong>
        </div>
        <div>
          <span>Sandbox size</span>
          <strong>
            {geometry.bounds_min.map((value, index) => {
              const max = geometry.bounds_max[index];
              if (value === null || max === null) {
                return null;
              }
              return Math.abs(max - value).toFixed(2);
            }).filter(Boolean).join(" x ")}
          </strong>
        </div>
        <div>
          <span>Display</span>
          <strong>{propertyMode === "material" ? "material" : isLoadingField ? "loading…" : propertyMode}</strong>
        </div>
        <div>
          <span>Snapshot</span>
          <strong>{selectedSnapshot ? selectedSnapshot.replace(".npz", "") : "-"}</strong>
        </div>
      </div>
    </div>
  );
}
