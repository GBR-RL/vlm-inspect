{{- define "vlm-inspect.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "vlm-inspect.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else if contains (include "vlm-inspect.name" .) .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "vlm-inspect.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "vlm-inspect.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
app.kubernetes.io/name: {{ include "vlm-inspect.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "vlm-inspect.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vlm-inspect.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "vlm-inspect.postgresql.fullname" -}}
{{- printf "%s-postgresql" (include "vlm-inspect.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Secret and key holding the SQLAlchemy database URL. */}}
{{- define "vlm-inspect.databaseSecret" -}}
{{- if .Values.postgresql.enabled }}
{{- include "vlm-inspect.postgresql.fullname" . }}
{{- else }}
{{- required "externalDatabase.existingSecret is required when postgresql.enabled=false" .Values.externalDatabase.existingSecret }}
{{- end }}
{{- end }}

{{- define "vlm-inspect.databaseSecretKey" -}}
{{- if .Values.postgresql.enabled }}database-url{{ else }}{{ .Values.externalDatabase.existingSecretKey }}{{ end }}
{{- end }}

{{- define "vlm-inspect.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}
