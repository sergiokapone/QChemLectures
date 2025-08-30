$source = "Syllabus_QChem.pdf"
$destination = "1к_маг_E6_Квантова_хімія_(ПО1)_Пономаренко_2025.pdf"

if (Test-Path $destination) {
    Remove-Item $destination -Force
}

Copy-Item -Path $source -Destination $destination