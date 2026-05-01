type LineCutPlotProps = {
  title?: string;
  message?: string;
};

export function LineCutPlot({
  title = "Line Cut Tools",
  message = "Line-cut and surface-cut visualization will plug into these endpoints next. The backend APIs are ready for staged frontend work.",
}: LineCutPlotProps) {
  return (
    <div className="viewer-block viewer-placeholder">
      <div className="viewer-block-header">
        <h5>{title}</h5>
        <span>coming next</span>
      </div>
      <p>{message}</p>
    </div>
  );
}
