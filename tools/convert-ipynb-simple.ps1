param(
    [Parameter(Mandatory = $true)]
    [string]$Notebook,

    [Parameter(Mandatory = $true)]
    [string]$HtmlOut
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Web

function Encode-Html {
    param([AllowNull()][object]$Value)
    if ($null -eq $Value) { return "" }
    return [System.Web.HttpUtility]::HtmlEncode(($Value -join ""))
}

function Join-Source {
    param([AllowNull()][object]$Source)
    if ($null -eq $Source) { return "" }
    if ($Source -is [array]) { return ($Source -join "") }
    return [string]$Source
}

function Render-Inline-Markdown {
    param([string]$Text)

    $mathPlaceholders = New-Object System.Collections.Generic.List[string]
    $textWithMathTokens = [regex]::Replace($Text, '\$(.+?)\$', {
        param($Match)
        $index = $mathPlaceholders.Count
        $mathPlaceholders.Add("<span class=""math-inline"">$(Convert-LatexToHtml $Match.Groups[1].Value)</span>")
        return "@@MATH$index@@"
    })

    $escaped = Encode-Html $textWithMathTokens
    $escaped = [regex]::Replace($escaped, '\*\*(.+?)\*\*', '<strong>$1</strong>')
    $escaped = [regex]::Replace($escaped, '\*(.+?)\*', '<em>$1</em>')
    $escaped = [regex]::Replace($escaped, '`([^`]+)`', '<code>$1</code>')

    for ($i = 0; $i -lt $mathPlaceholders.Count; $i++) {
        $escaped = $escaped.Replace("@@MATH$i@@", $mathPlaceholders[$i])
    }

    return $escaped
}

function Convert-LatexToHtml {
    param([string]$Latex)

    $html = Encode-Html ($Latex.Trim())
    $html = $html -replace '\\hat\{y\}_h', '<span class="math-hat">y</span><sub>h</sub>'
    $html = $html -replace '\\sigma_h', '&sigma;<sub>h</sub>'
    $html = $html -replace '\\pm', '&plusmn;'
    $html = $html -replace 'e_h', 'e<sub>h</sub>'
    $html = $html -replace 'y_h', 'y<sub>h</sub>'
    return $html
}

function Render-RawImage {
    param([string]$Line)

    if ($Line -notmatch '^\s*<img\s+([^>]+)>\s*$') { return $null }

    $attrs = $Matches[1]
    if ($attrs -notmatch 'src\s*=\s*["'']([^"'']+)["'']') { return $null }

    $src = $Matches[1]
    if ($src -match '^(https?:|data:|file:|/)') {
        $safeSrc = Encode-Html $src
    } else {
        $safeSrc = Encode-Html ("../" + $src.TrimStart("./"))
    }

    $width = ""
    if ($attrs -match 'width\s*=\s*["'']?([0-9]+%?)["'']?') {
        $widthValue = Encode-Html $Matches[1]
        $width = " width=""$widthValue"""
    }

    $alt = ""
    if ($attrs -match 'alt\s*=\s*["'']([^"'']+)["'']') {
        $alt = Encode-Html $Matches[1]
    }

    return "<p><img class=""markdown-image"" src=""$safeSrc""$width alt=""$alt"" /></p>"
}

function Render-Markdown {
    param([string]$Markdown)

    $lines = $Markdown -split "`r?`n"
    $html = New-Object System.Collections.Generic.List[string]
    $inList = $false
    $inFence = $false
    $inMath = $false
    $fence = New-Object System.Collections.Generic.List[string]
    $math = New-Object System.Collections.Generic.List[string]

    foreach ($line in $lines) {
        if ($line -match '^\s*\$\$\s*$') {
            if ($inMath) {
                $html.Add("<div class='math-display'>$(Convert-LatexToHtml ($math -join " "))</div>")
                $math.Clear()
                $inMath = $false
            } else {
                if ($inList) { $html.Add("</ul>"); $inList = $false }
                $inMath = $true
            }
            continue
        }

        if ($inMath) {
            $math.Add($line)
            continue
        }

        if ($line -match '^\s*```') {
            if ($inFence) {
                $html.Add("<pre><code>$(Encode-Html ($fence -join "`n"))</code></pre>")
                $fence.Clear()
                $inFence = $false
            } else {
                if ($inList) { $html.Add("</ul>"); $inList = $false }
                $inFence = $true
            }
            continue
        }

        if ($inFence) {
            $fence.Add($line)
            continue
        }

        if ($line -match '^\s*$') {
            if ($inList) { $html.Add("</ul>"); $inList = $false }
            continue
        }

        $rawImage = Render-RawImage $line
        if ($rawImage) {
            if ($inList) { $html.Add("</ul>"); $inList = $false }
            $html.Add($rawImage)
            continue
        }

        if ($line -match '^(#{1,6})\s+(.+)$') {
            if ($inList) { $html.Add("</ul>"); $inList = $false }
            $level = $Matches[1].Length
            $content = Render-Inline-Markdown $Matches[2]
            $html.Add("<h$level>$content</h$level>")
            continue
        }

        if ($line -match '^\s*[-*]\s+(.+)$') {
            if (-not $inList) { $html.Add("<ul>"); $inList = $true }
            $html.Add("<li>$(Render-Inline-Markdown $Matches[1])</li>")
            continue
        }

        if ($inList) { $html.Add("</ul>"); $inList = $false }
        $html.Add("<p>$(Render-Inline-Markdown $line)</p>")
    }

    if ($inFence) {
        $html.Add("<pre><code>$(Encode-Html ($fence -join "`n"))</code></pre>")
    }
    if ($inMath) {
        $html.Add("<div class='math-display'>$(Convert-LatexToHtml ($math -join " "))</div>")
    }
    if ($inList) { $html.Add("</ul>") }

    return ($html -join "`n")
}

function Get-DataValue {
    param(
        [AllowNull()][object]$Data,
        [string]$Name
    )
    if ($null -eq $Data) { return $null }
    $prop = $Data.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}

$notebookPath = (Resolve-Path -LiteralPath $Notebook).Path
$notebookRoot = Split-Path -Parent $notebookPath
$nb = Get-Content -Raw -LiteralPath $notebookPath | ConvertFrom-Json

$body = New-Object System.Collections.Generic.List[string]

foreach ($cell in $nb.cells) {
    if ($cell.cell_type -eq "markdown") {
        $body.Add("<section class='cell markdown'>$(Render-Markdown (Join-Source $cell.source))</section>")
        continue
    }

    if ($cell.cell_type -eq "code") {
        $source = Join-Source $cell.source
        $body.Add("<section class='cell code'><pre><code>$(Encode-Html $source)</code></pre>")

        foreach ($output in @($cell.outputs)) {
            if ($null -eq $output) { continue }

            if ($output.output_type -eq "stream") {
                $body.Add("<pre class='output text'>$(Encode-Html (Join-Source $output.text))</pre>")
                continue
            }

            if ($output.output_type -eq "error") {
                $trace = Join-Source $output.traceback
                $body.Add("<pre class='output error'>$(Encode-Html $trace)</pre>")
                continue
            }

            $data = $output.data
            $htmlValue = Get-DataValue $data "text/html"
            $pngValue = Get-DataValue $data "image/png"
            $jpegValue = Get-DataValue $data "image/jpeg"
            $svgValue = Get-DataValue $data "image/svg+xml"
            $plainValue = Get-DataValue $data "text/plain"

            if ($htmlValue) {
                $body.Add("<div class='output html'>$((Join-Source $htmlValue))</div>")
            } elseif ($pngValue) {
                $body.Add("<img class='output image' src='data:image/png;base64,$((Join-Source $pngValue).Trim())' />")
            } elseif ($jpegValue) {
                $body.Add("<img class='output image' src='data:image/jpeg;base64,$((Join-Source $jpegValue).Trim())' />")
            } elseif ($svgValue) {
                $body.Add("<div class='output svg'>$((Join-Source $svgValue))</div>")
            } elseif ($plainValue) {
                $body.Add("<pre class='output text'>$(Encode-Html (Join-Source $plainValue))</pre>")
            }
        }

        $body.Add("</section>")
    }
}

$title = [System.IO.Path]::GetFileNameWithoutExtension($notebookPath)
$css = @"
html { background: #f4f4f2; }
body {
  margin: 0 auto;
  max-width: 920px;
  padding: 36px 44px;
  background: #fff;
  color: #1f2933;
  font: 15px/1.55 "Segoe UI", Arial, sans-serif;
}
h1, h2, h3, h4 { color: #111827; line-height: 1.2; margin: 1.2em 0 0.45em; }
h1 { font-size: 30px; border-bottom: 1px solid #d7dce2; padding-bottom: 10px; }
h2 { font-size: 23px; border-bottom: 1px solid #eceff3; padding-bottom: 6px; }
h3 { font-size: 18px; }
p { margin: 0.55em 0; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }
th, td { border: 1px solid #d9dee5; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f2f5f8; }
.cell { page-break-inside: avoid; margin: 14px 0; }
.code pre {
  background: #f6f8fa;
  border: 1px solid #d8dee4;
  border-radius: 6px;
  overflow-x: auto;
  padding: 12px;
  font-size: 12px;
}
code, pre { font-family: Consolas, "Cascadia Mono", monospace; }
.math-inline, .math-display { font-family: Cambria Math, "Times New Roman", serif; }
.math-display { text-align: center; font-size: 19px; margin: 16px 0; }
.math-hat { position: relative; display: inline-block; padding-top: 0.1em; }
.math-hat::before { content: "^"; position: absolute; left: 0.05em; right: 0; top: -0.58em; text-align: center; font-size: 0.72em; }
.math-display sub, .math-inline sub { font-size: 0.72em; }
.output { margin: 10px 0 16px; }
.output.text, .output.error {
  white-space: pre-wrap;
  background: #fbfbfb;
  border-left: 4px solid #b8c2cc;
  padding: 10px 12px;
  font-size: 12px;
}
.output.error { border-left-color: #b91c1c; color: #7f1d1d; }
.output.image { display: block; max-width: 100%; height: auto; margin: 12px auto; }
img { max-width: 100%; }
.markdown-image { display: block; max-width: 100%; height: auto; margin: 12px auto; }
@page { margin: 16mm; }
@media print {
  html { background: #fff; }
  body { padding: 0; max-width: none; }
}
"@

$html = @"
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>$([System.Web.HttpUtility]::HtmlEncode($title))</title>
  <style>
$css
  </style>
</head>
<body>
$($body -join "`n")
</body>
</html>
"@

$outPath = Join-Path $notebookRoot $HtmlOut
$outDir = Split-Path -Parent $outPath
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

Set-Content -LiteralPath $outPath -Value $html -Encoding UTF8
Write-Host $outPath
