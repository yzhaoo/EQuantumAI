import { memo, useMemo } from "react";

type SnapshotTimelineProps = {
  snapshots: string[];
  activeSnapshot: string;
  onSelect: (snapshot: string) => void;
};

export const SnapshotTimeline = memo(function SnapshotTimeline({
  snapshots,
  activeSnapshot,
  onSelect,
}: SnapshotTimelineProps) {
  // Parsing node types
  const nodes = useMemo(() => {
    return snapshots.map((name) => {
      let type: "triangle" | "square" | "circle" | "star" | "colored-star" = "circle";
      const lower = name.toLowerCase();
      if (lower.includes("local")) {
        type = "triangle";
      } else if (lower.includes("poisson")) {
        type = "square";
      } else if (lower.includes("quantum")) {
        type = "circle";
      } else if (lower.includes("converged") || lower.includes("convered")) {
        type = "star";
      } else if (lower.includes("maxed")) {
        type = "colored-star";
      }
      return { name, type };
    });
  }, [snapshots]);

  return (
    <div className="snapshot-timeline-container">
      <div className="snapshot-timeline-scroll">
        <svg
          className="snapshot-timeline-svg"
          width={Math.max(nodes.length * 40 + 40, 400)}
          height={60}
          viewBox={`0 0 ${Math.max(nodes.length * 40 + 40, 400)} 60`}
        >
          {/* Connecting Line */}
          {nodes.length > 1 && (
            <line
              x1={20}
              y1={30}
              x2={20 + (nodes.length - 1) * 40}
              y2={30}
              stroke="#d1d5db"
              strokeWidth={2}
            />
          )}

          {/* Nodes */}
          {nodes.map((node, index) => {
            const x = 20 + index * 40;
            const y = 30;
            const isActive = node.name === activeSnapshot;
            const fill = isActive ? "#34d399" : "transparent";
            const stroke = "#4b5563";
            const strokeWidth = 2.5;

            let shape;
            if (node.type === "triangle") {
              const r = 8;
              shape = (
                <polygon
                  points={`${x},${y - r} ${x - r},${y + r * 0.7} ${x + r},${y + r * 0.7}`}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  strokeLinejoin="round"
                />
              );
            } else if (node.type === "square") {
              const size = 14;
              shape = (
                <rect
                  x={x - size / 2}
                  y={y - size / 2}
                  width={size}
                  height={size}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  rx={2}
                />
              );
            } else if (node.type === "star" || node.type === "colored-star") {
              const nodeFill = node.type === "colored-star" && !isActive ? "#f59e0b" : fill;
              const nodeStroke = node.type === "colored-star" && !isActive ? "#d97706" : stroke;
              shape = (
                <path
                  d={`M ${x},${y-9} L ${x+2.2},${y-2.8} L ${x+8.5},${y-2.8} L ${x+3.2},${y+1} L ${x+5.3},${y+7.3} L ${x},${y+3.4} L ${x-5.3},${y+7.3} L ${x-3.2},${y+1} L ${x-8.5},${y-2.8} L ${x-2.2},${y-2.8} Z`}
                  fill={nodeFill}
                  stroke={nodeStroke}
                  strokeWidth={strokeWidth}
                  strokeLinejoin="round"
                />
              );
            } else {
              shape = (
                <circle
                  cx={x}
                  cy={y}
                  r={7}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                />
              );
            }

            return (
              <g
                key={node.name}
                className="snapshot-timeline-node"
                onClick={() => onSelect(node.name)}
                style={{ cursor: "pointer" }}
              >
                {/* Invisible hit area */}
                <circle cx={x} cy={y} r={16} fill="transparent" />
                {shape}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="snapshot-timeline-footer">
        <div className="snapshot-timeline-legend">
          <span className="legend-item">
            <svg width={14} height={14} viewBox="0 0 14 14">
              <polygon points="7,1 1,11 13,11" fill="transparent" stroke="#4b5563" strokeWidth={2} strokeLinejoin="round" />
            </svg>
            Local solver
          </span>
          <span className="legend-item">
            <svg width={14} height={14} viewBox="0 0 14 14">
              <rect x="1" y="1" width="12" height="12" fill="transparent" stroke="#4b5563" strokeWidth={2} rx={2} />
            </svg>
            Poisson solver
          </span>
          <span className="legend-item">
            <svg width={14} height={14} viewBox="0 0 14 14">
              <circle cx="7" cy="7" r="6" fill="transparent" stroke="#4b5563" strokeWidth={2} />
            </svg>
            Quantum solver
          </span>
          <span className="legend-item">
            <svg width={16} height={16} viewBox="-10 -10 20 20">
              <path d="M 0,-9 L 2.2,-2.8 L 8.5,-2.8 L 3.2,1 L 5.3,7.3 L 0,3.4 L -5.3,7.3 L -3.2,1 L -8.5,-2.8 L -2.2,-2.8 Z" fill="transparent" stroke="#4b5563" strokeWidth={2} strokeLinejoin="round" />
            </svg>
            Converged
          </span>
          <span className="legend-item">
            <svg width={16} height={16} viewBox="-10 -10 20 20">
              <path d="M 0,-9 L 2.2,-2.8 L 8.5,-2.8 L 3.2,1 L 5.3,7.3 L 0,3.4 L -5.3,7.3 L -3.2,1 L -8.5,-2.8 L -2.2,-2.8 Z" fill="#f59e0b" stroke="#d97706" strokeWidth={2} strokeLinejoin="round" />
            </svg>
            Maxed
          </span>
        </div>
        <div className="snapshot-timeline-filename">
          File name: {activeSnapshot || "none"}
        </div>
      </div>
    </div>
  );
});
